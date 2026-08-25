"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams

from sglang_omni.scheduling.sglang_backend.evict_heap_radix_cache import (
    EvictHeapRadixCache,
)


def create_tree_cache(
    server_args,
    req_to_token_pool,
    token_to_kv_pool_allocator,
    page_size: int,
):
    """Create a tree cache based on server_args.

    When radix cache is disabled we always return ChunkCache so the scheduler
    keeps plain KV-cache semantics without any prefix matching.
    """
    params = CacheInitParams(
        disable=server_args.disable_radix_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        page_size=page_size,
        chunked_prefill_size=server_args.chunked_prefill_size,
    )

    import os

    if server_args.disable_radix_cache or (
        os.environ.get("SGLANG_OMNI_PROBE_DISABLE_RADIX", "").strip() == "1"
    ):
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        params.disable = True
        return ChunkCache(params)

    # PROBE ONLY (#1723/#1724 evidence): env-gated arm selection, all-unique
    # traffic emulation, and per-evict tracing. Never merge.
    from sglang.srt.mem_cache.radix_cache import RadixCache

    base_cls = (
        RadixCache
        if os.environ.get("SGLANG_OMNI_PROBE_BASELINE", "").strip() == "1"
        else EvictHeapRadixCache
    )

    if os.environ.get("SGLANG_OMNI_PROBE_NO_MATCH", "").strip() == "1":

        class _NoMatch(base_cls):
            def cache_finished_req(self, req, *args, **kwargs):
                import uuid

                try:
                    base = getattr(req, "extra_key", None)
                    req.extra_key = f"{base or ''}-salt-{uuid.uuid4().hex}"
                except Exception:
                    pass
                return super().cache_finished_req(req, *args, **kwargs)

            def match_prefix(self, *args, **kwargs):
                result = super().match_prefix(*args, **kwargs)
                try:
                    empty = result.device_indices[:0]
                    return type(result)(
                        device_indices=empty,
                        last_device_node=self.root_node,
                        last_host_node=self.root_node,
                    )
                except Exception:
                    return result

        base_cls = _NoMatch

    cache = base_cls(params)

    trace = os.environ.get("SGLANG_OMNI_PROBE_EVICT_TRACE", "").strip()
    if trace:
        import time as _time

        _f = open(trace, "a", buffering=1)
        _buf = []
        _orig_evict = cache.evict

        def _traced_evict(evict_params):
            t0 = _time.perf_counter()
            result = _orig_evict(evict_params)
            dt_us = (_time.perf_counter() - t0) * 1e6
            _buf.append(
                f"{_time.time():.3f},{dt_us:.1f},{len(cache.evictable_leaves)},"
                f"{len(getattr(cache, '_evict_heap', []))}"
            )
            if len(_buf) >= 200:
                _f.write("\n".join(_buf) + "\n")
                _buf.clear()
            return result

        cache.evict = _traced_evict

    return cache
