"""Tree cache factory using upstream SGLang CacheInitParams."""

from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache


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
        # PROBE ONLY (private #87): env gate so paired A/B arms run identical
        # code and differ only by this variable. Never merge.
        from sglang.srt.mem_cache.chunk_cache import ChunkCache

        params.disable = True
        return ChunkCache(params)

    if os.environ.get("SGLANG_OMNI_PROBE_NO_MATCH", "").strip() == "1":
        # PROBE ONLY (private #87): emulate production all-unique traffic —
        # every match misses, every finished request still inserts/retains.
        # Never merge.
        class _NoMatchRadixCache(RadixCache):
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

        return _NoMatchRadixCache(params)

    return RadixCache(params)
