"""Patched pretransform (reshape-based downsampling/upsampling) for MOSS Audio Tokenizer."""

from __future__ import annotations

import mlx.core as mx


class PatchedPretransform:
    """Reshape-based patching: merges/splits time steps and channels.

    Encode (downsample): (B, D, T) -> (B, D*patch, T//patch)
    Decode (upsample):   (B, D*patch, T) -> (B, D, T*patch)
    """

    def __init__(self, patch_size: int, is_downsample: bool):
        self.patch_size = patch_size
        self.is_downsample = is_downsample
        self.downsample_ratio = patch_size

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_downsample:
            return self.encode(x)
        else:
            return self.decode(x)

    def encode(self, x: mx.array) -> mx.array:
        """(B, D, T) -> (B, D*patch, T//patch)"""
        B, D, T = x.shape
        h = self.patch_size
        # Reshape: (B, D, T//h, h) -> permute to (B, D, h, T//h) -> (B, D*h, T//h)
        x = x.reshape(B, D, -1, h)        # (B, D, T//h, h)
        x = x.transpose(0, 1, 3, 2)       # (B, D, h, T//h)
        x = x.reshape(B, D * h, -1)       # (B, D*h, T//h)
        return x

    def decode(self, x: mx.array) -> mx.array:
        """(B, D*patch, T) -> (B, D, T*patch)"""
        B, DH, L = x.shape
        h = self.patch_size
        D = DH // h
        # Reshape: (B, D, h, L) -> permute to (B, D, L, h) -> (B, D, L*h)
        x = x.reshape(B, D, h, L)         # (B, D, h, L)
        x = x.transpose(0, 1, 3, 2)       # (B, D, L, h)
        x = x.reshape(B, D, L * h)        # (B, D, L*h)
        return x

    def parameters(self):
        return {}

    def update(self, params):
        pass
