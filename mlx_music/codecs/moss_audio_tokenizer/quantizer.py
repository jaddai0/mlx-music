"""Vector quantization modules for MOSS Audio Tokenizer (MLX)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# Weight-Normalized Conv1d (kernel_size=1, equivalent to weight-normed Linear)
# =============================================================================


class WNConv1d(nn.Module):
    """Weight-normalized 1D convolution with kernel_size=1 (equivalent to Linear)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # weight_v: (out, 1, in) for conv1d format, but we store as (out, in)
        # and apply as matmul for efficiency since kernel_size=1
        k = 1.0 / (in_channels ** 0.5)
        self.weight_v = mx.random.uniform(-k, k, (out_channels, in_channels))
        self.weight_g = mx.ones((out_channels, 1))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, C_in, T) -> (B, C_out, T)"""
        # Compute normalized weight: w = g * v / ||v||
        v_norm = mx.sqrt(mx.sum(self.weight_v * self.weight_v, axis=1, keepdims=True) + 1e-12)
        weight = self.weight_g * self.weight_v / v_norm

        # (B, C_in, T) -> (B, T, C_in) @ (C_in, C_out) -> (B, T, C_out) -> (B, C_out, T)
        x = x.transpose(0, 2, 1)
        x = x @ weight.T + self.bias
        return x.transpose(0, 2, 1)


# =============================================================================
# Lookup-Free Quantization (LFQ)
# =============================================================================


class LFQ(nn.Module):
    """Single-codebook Lookup-Free Quantization (inference only)."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if input_dim != codebook_dim:
            self.in_proj = WNConv1d(input_dim, codebook_dim)
            self.out_proj = WNConv1d(codebook_dim, input_dim)
        else:
            self.in_proj = None
            self.out_proj = None

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices: mx.array) -> mx.array:
        """Decode code indices to embeddings. indices: (B, T) -> (B, D, T)"""
        z_q = self.codebook(indices)  # (B, T, codebook_dim)
        z_q = z_q.transpose(0, 2, 1)  # (B, codebook_dim, T)
        if self.out_proj is not None:
            z_q = self.out_proj(z_q)
        return z_q

    def encode(self, z: mx.array) -> tuple[mx.array, mx.array]:
        """Encode input to nearest codes. z: (B, D, T) -> (z_q, indices)"""
        if self.in_proj is not None:
            z_e = self.in_proj(z)
        else:
            z_e = z

        # (B, D, T) -> (B*T, D)
        B, D, T = z_e.shape
        encodings = z_e.transpose(0, 2, 1).reshape(-1, D)
        codebook = self.codebook.weight

        # L2 normalize for LFQ
        enc_norm = mx.sqrt(mx.sum(encodings * encodings, axis=1, keepdims=True) + 1e-12)
        encodings = encodings / enc_norm
        cb_norm = mx.sqrt(mx.sum(codebook * codebook, axis=1, keepdims=True) + 1e-12)
        codebook_n = codebook / cb_norm

        # Squared distance
        dist = (
            mx.sum(encodings * encodings, axis=1, keepdims=True)
            - 2 * (encodings @ codebook_n.T)
            + mx.sum(codebook_n * codebook_n, axis=1, keepdims=True).T
        )
        indices = mx.argmin(dist, axis=1)
        indices = indices.reshape(B, T)

        # Decode
        z_q = self.codebook(indices).transpose(0, 2, 1)  # (B, D, T)
        return z_q, indices


# =============================================================================
# Vector Quantize (standard)
# =============================================================================


class VectorQuantize(nn.Module):
    """Single-codebook VQ (inference only)."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if input_dim != codebook_dim:
            self.in_proj = WNConv1d(input_dim, codebook_dim)
            self.out_proj = WNConv1d(codebook_dim, input_dim)
        else:
            self.in_proj = None
            self.out_proj = None

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices: mx.array) -> mx.array:
        """indices: (B, T) -> (B, D, T)"""
        z_q = self.codebook(indices).transpose(0, 2, 1)
        if self.out_proj is not None:
            z_q = self.out_proj(z_q)
        return z_q

    def encode(self, z: mx.array) -> tuple[mx.array, mx.array]:
        """z: (B, D, T) -> (z_q, indices)"""
        if self.in_proj is not None:
            z_e = self.in_proj(z)
        else:
            z_e = z

        B, D, T = z_e.shape
        encodings = z_e.transpose(0, 2, 1).reshape(-1, D)
        codebook = self.codebook.weight

        dist = (
            mx.sum(encodings * encodings, axis=1, keepdims=True)
            - 2 * (encodings @ codebook.T)
            + mx.sum(codebook * codebook, axis=1, keepdims=True).T
        )
        indices = mx.argmin(dist, axis=1).reshape(B, T)
        z_q = self.codebook(indices).transpose(0, 2, 1)
        return z_q, indices


# =============================================================================
# Residual Quantizers
# =============================================================================


class ResidualLFQ(nn.Module):
    """Residual Lookup-Free Quantization with multiple codebooks."""

    def __init__(
        self,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers

        if input_dim != self.rvq_dim:
            self.input_proj = WNConv1d(input_dim, self.rvq_dim)
        else:
            self.input_proj = None

        if self.rvq_dim != self.output_dim:
            self.output_proj = WNConv1d(self.rvq_dim, self.output_dim)
        else:
            self.output_proj = None

        self.quantizers = [
            LFQ(
                input_dim=self.rvq_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        ]

    def decode_codes(self, codes: mx.array) -> mx.array:
        """Decode stacked codes to embeddings.

        codes: (N_q, B, T) -> (B, output_dim, T)
        """
        nq = codes.shape[0]
        B, T = codes.shape[1], codes.shape[2]
        emb = mx.zeros((B, self.rvq_dim, T))

        for i in range(min(nq, self.num_quantizers)):
            emb = emb + self.quantizers[i].decode_code(codes[i])

        if self.output_proj is not None:
            emb = self.output_proj(emb)
        return emb

    def encode(self, z: mx.array, n_quantizers: int | None = None) -> tuple[mx.array, mx.array]:
        """z: (B, D, T) -> (quantized, all_indices: (N_q, B, T))"""
        if self.input_proj is not None:
            z = self.input_proj(z)

        quantized = mx.zeros_like(z)
        residual = z
        all_indices = []

        nq = n_quantizers or self.num_quantizers
        for i in range(min(nq, self.num_quantizers)):
            z_q_i, indices_i = self.quantizers[i].encode(residual)
            quantized = quantized + z_q_i
            residual = residual - z_q_i
            all_indices.append(indices_i)

        all_indices = mx.stack(all_indices, axis=0)  # (N_q, B, T)

        if self.output_proj is not None:
            quantized = self.output_proj(quantized)
        return quantized, all_indices


class ResidualVQ(nn.Module):
    """Residual Vector Quantization with multiple codebooks."""

    def __init__(
        self,
        input_dim: int = 1024,
        rvq_dim: int | None = None,
        output_dim: int | None = None,
        num_quantizers: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rvq_dim = rvq_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.num_quantizers = num_quantizers

        if input_dim != self.rvq_dim:
            self.input_proj = WNConv1d(input_dim, self.rvq_dim)
        else:
            self.input_proj = None

        if self.rvq_dim != self.output_dim:
            self.output_proj = WNConv1d(self.rvq_dim, self.output_dim)
        else:
            self.output_proj = None

        self.quantizers = [
            VectorQuantize(
                input_dim=self.rvq_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
            )
            for _ in range(num_quantizers)
        ]

    def decode_codes(self, codes: mx.array) -> mx.array:
        """codes: (N_q, B, T) -> (B, output_dim, T)"""
        nq = codes.shape[0]
        B, T = codes.shape[1], codes.shape[2]
        emb = mx.zeros((B, self.rvq_dim, T))

        for i in range(min(nq, self.num_quantizers)):
            emb = emb + self.quantizers[i].decode_code(codes[i])

        if self.output_proj is not None:
            emb = self.output_proj(emb)
        return emb
