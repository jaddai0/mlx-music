"""
Tests for mlx_music.models.musicgen.kv_cache module.

Tests:
- KVCache: update_and_fetch, reset, state, seq_len
- CrossAttentionCache: set, get, is_set, reset
- MusicGenCacheState: dataclass, to_legacy_format, from_legacy_format
- create_causal_mask: shape, values, past_len handling
"""

import pytest
import mlx.core as mx


class TestKVCache:
    """Tests for KVCache class."""

    def test_init_empty(self):
        """KVCache should initialize with no cached values."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0
        assert cache.seq_len == 0

    def test_update_and_fetch_first_call(self):
        """First update should allocate buffer and return values."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))

        all_keys, all_values = cache.update_and_fetch(keys, values)
        mx.synchronize()

        assert all_keys.shape == (2, 10, 64)
        assert all_values.shape == (2, 10, 64)
        assert cache.offset == 10

    def test_update_and_fetch_accumulates(self):
        """Multiple updates should accumulate cached values."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        # First update
        k1 = mx.random.normal((2, 10, 64))
        v1 = mx.random.normal((2, 10, 64))
        cache.update_and_fetch(k1, v1)

        # Second update
        k2 = mx.random.normal((2, 5, 64))
        v2 = mx.random.normal((2, 5, 64))
        all_keys, all_values = cache.update_and_fetch(k2, v2)
        mx.synchronize()

        # Should now have 15 total cached
        assert all_keys.shape == (2, 15, 64)
        assert all_values.shape == (2, 15, 64)
        assert cache.offset == 15

    def test_reset_clears_cache(self):
        """reset() should clear all cached values."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        # Add some values
        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))
        cache.update_and_fetch(keys, values)

        # Reset
        cache.reset()

        assert cache.keys is None
        assert cache.values is None
        assert cache.offset == 0

    def test_state_property_returns_valid_portion(self):
        """state property should return only valid (used) portion of cache."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))
        cache.update_and_fetch(keys, values)

        k, v = cache.state
        mx.synchronize()

        assert k.shape == (2, 10, 64)
        assert v.shape == (2, 10, 64)

    def test_state_property_empty_cache(self):
        """state property should return (None, None) for empty cache."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        k, v = cache.state
        assert k is None
        assert v is None

    def test_seq_len_property(self):
        """seq_len should return current cached sequence length."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)
        assert cache.seq_len == 0

        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))
        cache.update_and_fetch(keys, values)

        assert cache.seq_len == 10

    def test_buffer_grows_with_step(self):
        """Buffer should grow by step size when needed."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        step = 32
        cache = KVCache(embed_dim=64, step=step)

        # Add values that exceed initial step
        keys = mx.random.normal((2, step + 10, 64))
        values = mx.random.normal((2, step + 10, 64))
        cache.update_and_fetch(keys, values)

        # Buffer should have grown to accommodate
        assert cache.keys.shape[1] >= step + 10


class TestCrossAttentionCache:
    """Tests for CrossAttentionCache class."""

    def test_init_empty(self):
        """CrossAttentionCache should initialize empty."""
        from mlx_music.models.musicgen.kv_cache import CrossAttentionCache

        cache = CrossAttentionCache()

        assert cache.keys is None
        assert cache.values is None
        assert cache.is_set is False

    def test_set_stores_values(self):
        """set() should store keys and values."""
        from mlx_music.models.musicgen.kv_cache import CrossAttentionCache

        cache = CrossAttentionCache()

        keys = mx.random.normal((2, 50, 64))
        values = mx.random.normal((2, 50, 64))

        cache.set(keys, values)

        assert cache.is_set is True
        assert cache.keys is not None
        assert cache.values is not None

    def test_get_returns_cached_values(self):
        """get() should return cached keys and values."""
        from mlx_music.models.musicgen.kv_cache import CrossAttentionCache

        cache = CrossAttentionCache()

        keys = mx.random.normal((2, 50, 64))
        values = mx.random.normal((2, 50, 64))
        cache.set(keys, values)

        k, v = cache.get()
        mx.synchronize()

        assert mx.allclose(k, keys)
        assert mx.allclose(v, values)

    def test_get_empty_returns_none(self):
        """get() on empty cache should return (None, None)."""
        from mlx_music.models.musicgen.kv_cache import CrossAttentionCache

        cache = CrossAttentionCache()

        k, v = cache.get()

        assert k is None
        assert v is None

    def test_reset_clears_cache(self):
        """reset() should clear cached values."""
        from mlx_music.models.musicgen.kv_cache import CrossAttentionCache

        cache = CrossAttentionCache()

        keys = mx.random.normal((2, 50, 64))
        values = mx.random.normal((2, 50, 64))
        cache.set(keys, values)

        cache.reset()

        assert cache.is_set is False
        assert cache.keys is None
        assert cache.values is None


class TestMusicGenCacheState:
    """Tests for MusicGenCacheState dataclass."""

    def test_init_creates_caches(self):
        """MusicGenCacheState should create caches for all layers."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState

        state = MusicGenCacheState(num_layers=12, embed_dim=64)

        assert len(state.self_attn_caches) == 12
        assert len(state.cross_attn_caches) == 12

    def test_get_self_attn_cache(self):
        """get_self_attn_cache should return cache for layer."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState, KVCache

        state = MusicGenCacheState(num_layers=12, embed_dim=64)

        cache = state.get_self_attn_cache(5)

        assert isinstance(cache, KVCache)

    def test_get_cross_attn_cache(self):
        """get_cross_attn_cache should return cache for layer."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState, CrossAttentionCache

        state = MusicGenCacheState(num_layers=12, embed_dim=64)

        cache = state.get_cross_attn_cache(5)

        assert isinstance(cache, CrossAttentionCache)

    def test_reset_clears_all_caches(self):
        """reset() should clear all caches."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState

        state = MusicGenCacheState(num_layers=4, embed_dim=64)

        # Populate some caches
        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))
        state.self_attn_caches[0].update_and_fetch(keys, values)
        state.cross_attn_caches[0].set(keys, values)

        # Reset
        state.reset()

        assert state.self_attn_caches[0].offset == 0
        assert state.cross_attn_caches[0].is_set is False

    def test_past_seq_len_property(self):
        """past_seq_len should return cached self-attention length."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState

        state = MusicGenCacheState(num_layers=4, embed_dim=64)

        assert state.past_seq_len == 0

        keys = mx.random.normal((2, 15, 64))
        values = mx.random.normal((2, 15, 64))
        state.self_attn_caches[0].update_and_fetch(keys, values)

        assert state.past_seq_len == 15

    def test_cross_attn_ready_property(self):
        """cross_attn_ready should check if all cross-attention caches are set."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState

        state = MusicGenCacheState(num_layers=4, embed_dim=64)

        assert state.cross_attn_ready is False

        keys = mx.random.normal((2, 50, 64))
        values = mx.random.normal((2, 50, 64))

        # Set only some
        state.cross_attn_caches[0].set(keys, values)
        state.cross_attn_caches[1].set(keys, values)

        assert state.cross_attn_ready is False

        # Set all
        state.cross_attn_caches[2].set(keys, values)
        state.cross_attn_caches[3].set(keys, values)

        assert state.cross_attn_ready is True

    def test_to_legacy_format(self):
        """to_legacy_format should convert to tuple-based format."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState

        state = MusicGenCacheState(num_layers=2, embed_dim=64)

        # Populate caches
        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))
        state.self_attn_caches[0].update_and_fetch(keys, values)
        state.cross_attn_caches[0].set(keys, values)

        past_kv, cross_kv = state.to_legacy_format()

        assert len(past_kv) == 2
        assert len(cross_kv) == 2
        # First layer should have data
        assert past_kv[0] is not None
        assert cross_kv[0] is not None
        # Second layer should be None
        assert past_kv[1] is None
        assert cross_kv[1] is None

    def test_from_legacy_format(self):
        """from_legacy_format should create state from tuples."""
        from mlx_music.models.musicgen.kv_cache import MusicGenCacheState

        keys = mx.random.normal((2, 10, 64))
        values = mx.random.normal((2, 10, 64))

        past_kv = [(keys, values), None]
        cross_kv = [None, (keys, values)]

        state = MusicGenCacheState.from_legacy_format(
            past_kv, cross_kv, num_layers=2, embed_dim=64
        )

        # Check self-attention cache was restored
        assert state.self_attn_caches[0].offset == 10
        assert state.self_attn_caches[1].offset == 0

        # Check cross-attention cache was restored
        assert state.cross_attn_caches[0].is_set is False
        assert state.cross_attn_caches[1].is_set is True


class TestCreateCausalMask:
    """Tests for create_causal_mask function."""

    def test_causal_mask_shape(self):
        """create_causal_mask should return correct shape."""
        from mlx_music.models.musicgen.kv_cache import create_causal_mask

        mask = create_causal_mask(seq_len=10, past_len=0)

        assert mask.shape == (1, 1, 10, 10)

    def test_causal_mask_shape_with_past(self):
        """create_causal_mask with past should have extended key dimension."""
        from mlx_music.models.musicgen.kv_cache import create_causal_mask

        mask = create_causal_mask(seq_len=5, past_len=10)

        # Query dim = 5, Key dim = 15 (past + current)
        assert mask.shape == (1, 1, 5, 15)

    def test_causal_mask_values_no_past(self):
        """Causal mask without past should be lower triangular."""
        from mlx_music.models.musicgen.kv_cache import create_causal_mask

        mask = create_causal_mask(seq_len=4, past_len=0)
        mx.synchronize()

        # Row 0 can attend to position 0 only
        assert mask[0, 0, 0, 0] == 0.0
        assert mask[0, 0, 0, 1] == float("-inf")

        # Row 3 can attend to positions 0-3
        assert mask[0, 0, 3, 0] == 0.0
        assert mask[0, 0, 3, 3] == 0.0

    def test_causal_mask_values_with_past(self):
        """Causal mask with past should allow attending to all past."""
        from mlx_music.models.musicgen.kv_cache import create_causal_mask

        mask = create_causal_mask(seq_len=2, past_len=5)
        mx.synchronize()

        # Row 0 (first new token) can attend to past (0-4) and itself (5)
        for i in range(6):
            assert mask[0, 0, 0, i] == 0.0
        # But not future
        assert mask[0, 0, 0, 6] == float("-inf")

        # Row 1 (second new token) can attend to past and both new tokens
        for i in range(7):
            assert mask[0, 0, 1, i] == 0.0

    def test_causal_mask_float_type(self):
        """Causal mask should have float values."""
        from mlx_music.models.musicgen.kv_cache import create_causal_mask

        mask = create_causal_mask(seq_len=4, past_len=0)

        assert mask.dtype == mx.float32


class TestKVCacheIntegration:
    """Integration tests for KV cache in generation scenarios."""

    def test_autoregressive_generation_simulation(self):
        """Simulate autoregressive generation with cache."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        # Simulate generating 10 tokens one at a time
        for step in range(10):
            # Generate one token at a time
            k = mx.random.normal((1, 1, 64))
            v = mx.random.normal((1, 1, 64))

            all_k, all_v = cache.update_and_fetch(k, v)
            mx.synchronize()

            # Should accumulate properly
            expected_len = step + 1
            assert all_k.shape == (1, expected_len, 64)
            assert all_v.shape == (1, expected_len, 64)

    def test_prefill_then_decode_simulation(self):
        """Simulate prefill (prompt) then decode (generation)."""
        from mlx_music.models.musicgen.kv_cache import KVCache

        cache = KVCache(embed_dim=64, step=256)

        # Prefill: process all prompt tokens at once
        prompt_len = 50
        k_prompt = mx.random.normal((1, prompt_len, 64))
        v_prompt = mx.random.normal((1, prompt_len, 64))

        all_k, all_v = cache.update_and_fetch(k_prompt, v_prompt)
        mx.synchronize()
        assert all_k.shape == (1, prompt_len, 64)

        # Decode: generate tokens one at a time
        for i in range(10):
            k = mx.random.normal((1, 1, 64))
            v = mx.random.normal((1, 1, 64))

            all_k, all_v = cache.update_and_fetch(k, v)
            mx.synchronize()

            expected_len = prompt_len + i + 1
            assert all_k.shape == (1, expected_len, 64)
