# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (env-gated): capture one target forward of Fun-CosyVoice3.

Records, for the first forward whose batch matches the trigger, the inputs
(ids, embedding rows, positions, attention metadata), the prompt KV rows read
from the pool, per-module outputs at every row (forward hooks; cloned at hook
time because fused add-RMSNorm updates buffers in place) and the full logits.
Used to diff a non-speculative DECODE against a TARGET_VERIFY row.

Env: SGLANG_OMNI_DIVPROBE_DIR (output dir; unset = disabled),
     SGLANG_OMNI_DIVPROBE_POS (query position, default 131),
     SGLANG_OMNI_DIVPROBE_ARM (label), SGLANG_OMNI_DIVPROBE_MAX (per kind, default 1).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import torch

_DIR = os.environ.get("SGLANG_OMNI_DIVPROBE_DIR")
_POS = int(os.environ.get("SGLANG_OMNI_DIVPROBE_POS", "131"))
_ARM = os.environ.get("SGLANG_OMNI_DIVPROBE_ARM", "arm")
_MAX = int(os.environ.get("SGLANG_OMNI_DIVPROBE_MAX", "1"))
_counts: dict[str, int] = {}

ENABLED = bool(_DIR)


def _bits(t: torch.Tensor) -> np.ndarray:
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).cpu().numpy()
    if t.dtype in (torch.int32, torch.int64, torch.bool):
        return t.cpu().numpy()
    return t.float().cpu().numpy()


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _list(t: Any, limit: int = 4096) -> Any:
    if t is None:
        return None
    if torch.is_tensor(t):
        if t.numel() > limit:
            return {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sha": _sha(_bits(t)),
            }
        return t.detach().cpu().tolist()
    if isinstance(t, (list, tuple)):
        return [_list(x, limit) for x in t]
    if isinstance(t, (int, float, str, bool)) or t is None:
        return t
    return repr(t)[:200]


def match(forward_batch: Any, positions: torch.Tensor) -> str | None:
    """Return a capture kind for this forward or None."""
    if not ENABLED:
        return None
    try:
        if int(forward_batch.batch_size) != 1:
            return None
        mode = forward_batch.forward_mode
        if (
            mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend()
        ):
            kind = "prefill" if int(forward_batch.seq_lens[0]) == _POS else None
        else:
            pos = positions.detach().cpu().tolist()
            kind = f"q-{mode.name}" if _POS in pos else None
        if kind is None or _counts.get(kind, 0) >= _MAX:
            return None
        _counts[kind] = _counts.get(kind, 0) + 1
        return kind
    except Exception:  # pragma: no cover - probe only
        return None


class Hooks:
    def __init__(self, backbone: torch.nn.Module):
        self.store: dict[str, Any] = {}
        self.handles: list[Any] = []
        st = self.store

        def _cl(x):
            if torch.is_tensor(x):
                return x.detach().clone()
            if isinstance(x, tuple):
                return tuple(_cl(p) for p in x)
            return x

        def mk(name):
            def hook(_m, _args, out):
                st[name] = _cl(out)

            return hook

        H = self.handles
        H.append(
            backbone.layers[0].register_forward_pre_hook(
                lambda _m, args: st.__setitem__("layer0_in", _cl(args[1]))
            )
        )
        for i, layer in enumerate(backbone.layers):
            p = f"L{i:02d}"
            H.append(layer.register_forward_hook(mk(p)))
            H.append(layer.input_layernorm.register_forward_hook(mk(f"{p}.ln1")))
            H.append(layer.self_attn.qkv_proj.register_forward_hook(mk(f"{p}.qkv")))
            H.append(layer.self_attn.attn.register_forward_hook(mk(f"{p}.attn_core")))
            H.append(layer.self_attn.register_forward_hook(mk(f"{p}.attn_o")))
            H.append(layer.mlp.register_forward_hook(mk(f"{p}.mlp")))
        H.append(backbone.norm.register_forward_hook(mk("final_norm")))

    def remove(self):
        for h in self.handles:
            h.remove()

    def flat(self) -> dict[str, np.ndarray]:
        out = {}
        for name, val in self.store.items():
            if isinstance(val, tuple):
                for k, part in enumerate(val):
                    if torch.is_tensor(part):
                        out[f"{name}.{k}"] = _bits(part)
            elif torch.is_tensor(val):
                out[name] = _bits(val)
        return out


def _metadata_dict(md: Any) -> dict:
    out = {}
    if md is None:
        return out
    for k, v in vars(md).items():
        if torch.is_tensor(v):
            out[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "values": _list(v, 2048),
            }
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = repr(v)[:200]
    return out


def _kv_pool(forward_batch: Any, attn_backend: Any):
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    if pool is None:
        pool = getattr(attn_backend, "token_to_kv_pool", None)
    if pool is None:
        pool = attn_backend.model_runner.token_to_kv_pool
    return pool


def _kv_rows(pool: Any, backbone: torch.nn.Module, idx: torch.Tensor):
    ks, vs = [], []
    for layer in backbone.layers:
        lid = layer.self_attn.attn.layer_id
        ks.append(pool.get_key_buffer(lid)[idx])
        vs.append(pool.get_value_buffer(lid)[idx])
    return _bits(torch.stack(ks)), _bits(torch.stack(vs))


def save(
    kind: str,
    model: Any,
    hooks: Hooks,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: Any,
    input_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    logits_output: Any,
) -> None:
    t0 = time.time()
    torch.cuda.synchronize()
    os.makedirs(_DIR, exist_ok=True)
    tag = f"{_ARM}-{kind}-pos{_POS}"
    backbone = model.model
    rp = int(forward_batch.req_pool_indices[0].item())
    seq_len = int(forward_batch.seq_lens[0].item())
    from sglang.srt.model_executor.forward_context import get_attn_backend

    attn_backend = get_attn_backend()
    r2t_pool = getattr(forward_batch, "req_to_token_pool", None)
    if r2t_pool is None:
        r2t_pool = getattr(attn_backend, "req_to_token_pool", None)
    if r2t_pool is None:
        r2t_pool = attn_backend.model_runner.req_to_token_pool
    req_to_token = r2t_pool.req_to_token
    # prompt rows 0..POS-1 (for prefill: everything written by this forward)
    n_hist = seq_len if kind == "prefill" else min(seq_len, _POS)
    hist_idx = req_to_token[rp, :n_hist].to(torch.int64)
    pool = _kv_pool(forward_batch, attn_backend)
    k, v = _kv_rows(pool, backbone, hist_idx)
    out_loc = forward_batch.out_cache_loc
    qk = qv = None
    if out_loc is not None and out_loc.numel() > 0:
        qk, qv = _kv_rows(pool, backbone, out_loc.to(torch.int64))
    spec = forward_batch.spec_info
    md = getattr(attn_backend, "forward_metadata", None)
    md_expand = getattr(attn_backend, "forward_metadata_spec_decode_expand", None)
    logits = logits_output.next_token_logits
    lg = logits.float()
    top_v, top_i = lg.topk(5, dim=-1)
    pos_list = positions.detach().cpu().tolist()
    row = pos_list.index(_POS) if _POS in pos_list else None
    meta = dict(
        arm=_ARM,
        kind=kind,
        tag=tag,
        mode=str(forward_batch.forward_mode),
        row=row,
        position=_POS,
        batch_size=int(forward_batch.batch_size),
        num_rows=int(input_ids.shape[0]),
        input_ids=_list(input_ids),
        positions=pos_list,
        fb_positions=_list(forward_batch.positions),
        seq_lens=_list(forward_batch.seq_lens),
        seq_lens_cpu=_list(getattr(forward_batch, "seq_lens_cpu", None)),
        seq_lens_sum=getattr(forward_batch, "seq_lens_sum", None),
        extend_seq_lens=_list(getattr(forward_batch, "extend_seq_lens", None)),
        extend_prefix_lens=_list(getattr(forward_batch, "extend_prefix_lens", None)),
        extend_start_loc=_list(getattr(forward_batch, "extend_start_loc", None)),
        out_cache_loc=_list(out_loc),
        req_pool_indices=_list(forward_batch.req_pool_indices),
        req_to_token_row=req_to_token[rp, : seq_len + 12].cpu().tolist(),
        hist_idx_sha=_sha(hist_idx.cpu().numpy()),
        n_hist=n_hist,
        capture_hidden_mode=str(getattr(forward_batch, "capture_hidden_mode", None)),
        fb_input_embeds=(
            None
            if forward_batch.input_embeds is None
            else list(forward_batch.input_embeds.shape)
        ),
        mm_input_embeds=(
            None
            if getattr(forward_batch, "mm_input_embeds", None) is None
            else list(forward_batch.mm_input_embeds.shape)
        ),
        embeds_shape=list(input_embeds.shape),
        embeds_dtype=str(input_embeds.dtype),
        embeds_row_sha=[
            _sha(_bits(input_embeds[i])) for i in range(min(input_embeds.shape[0], 512))
        ],
        embeds_from_speech_table=[
            (
                bool(
                    torch.equal(
                        input_embeds[i],
                        model.speech_embedding.weight[int(input_ids[i])],
                    )
                )
                if 0 <= int(input_ids[i]) < model.speech_embedding.weight.shape[0]
                else None
            )
            for i in range(min(input_ids.shape[0], 16))
        ],
        attn_backend=type(attn_backend).__name__,
        attn_metadata=_metadata_dict(md),
        attn_metadata_expand=_metadata_dict(md_expand),
        kv_pool=type(pool).__name__,
        kv_k_sha_per_layer=[_sha(k[l]) for l in range(k.shape[0])],
        kv_v_sha_per_layer=[_sha(v[l]) for l in range(v.shape[0])],
        capture_aux=bool(getattr(model, "capture_aux_hidden_states", False)),
        layers_to_capture=getattr(backbone, "layers_to_capture", None),
        spec_info=None,
        sampling_info=(
            {
                "temperatures": _list(
                    getattr(forward_batch.sampling_info, "temperatures", None)
                ),
                "scaling": _list(
                    getattr(forward_batch.sampling_info, "acc_scaling_penalties", None)
                ),
            }
            if getattr(forward_batch, "sampling_info", None) is not None
            else None
        ),
        top_ids=top_i.cpu().tolist(),
        top_values=top_v.cpu().tolist(),
        logits_shape=list(logits.shape),
        hidden_shape=list(hidden_states.shape),
        captured_keys=sorted(hooks.store.keys()),
    )
    if spec is not None:
        meta["spec_info"] = dict(
            cls=type(spec).__name__,
            draft_token_num=getattr(spec, "draft_token_num", None),
            draft_token=_list(getattr(spec, "draft_token", None)),
            positions=_list(getattr(spec, "positions", None)),
            custom_mask=_list(getattr(spec, "custom_mask", None), 64),
            retrieve_index=_list(getattr(spec, "retrieve_index", None)),
            attrs=[a for a in vars(spec).keys()] if hasattr(spec, "__dict__") else None,
        )
    arrays = dict(
        kv_indices=hist_idx.cpu().numpy(),
        kv_k=k,
        kv_v=v,
        embeds=_bits(input_embeds),
        input_ids=_bits(input_ids),
        positions=_bits(positions),
        logits=_bits(logits),
        hidden_states=_bits(hidden_states),
    )
    if qk is not None:
        arrays["q_kv_k"] = qk
        arrays["q_kv_v"] = qv
    arrays.update(hooks.flat())
    np.savez_compressed(os.path.join(_DIR, tag + ".npz"), **arrays)
    meta["capture_seconds"] = time.time() - t0
    with open(os.path.join(_DIR, tag + ".json"), "w") as f:
        json.dump(meta, f)
    print(
        f"[divprobe] {tag} mode={meta['mode']} row={row} rows={meta['num_rows']} "
        f"ids={meta['input_ids'] if isinstance(meta['input_ids'], list) and len(meta['input_ids']) < 16 else '...'} "
        f"top={[(i[:3], [round(x, 3) for x in vv[:3]]) for i, vv in zip(meta['top_ids'][:2], meta['top_values'][:2])]} "
        f"in {meta['capture_seconds']:.1f}s",
        flush=True,
    )
