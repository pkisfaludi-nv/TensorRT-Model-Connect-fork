# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2 attention, RoPE and feed-forward lowering shared by the connectors and the DiT.

Mirrors diffusers ``LTX2Attention`` + ``LTX2AudioVideoAttnProcessor`` /
``LTX2PerturbedAttnProcessor`` and ``apply_split_rotary_emb``:

- Q/K RMSNorm spans all heads of a token (``rms_norm_across_heads``).
- Split RoPE rotates the two halves of every head:
  ``out1 = x1*cos - x2*sin``, ``out2 = x2*cos + x1*sin`` in fp32, applied
  elementwise on a ``[B, T, H, 2, r]`` view (no dense rotate-half matrix).
- Gated attention multiplies each head's context by ``2*sigmoid(to_gate_logits(x_q))``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorrt as trt

from .graph import Graph


@dataclass
class RopeTables:
    """Split-RoPE cos/sin tensors of shape ``[1, T, H, 1, r]`` (fp32, already in the network)."""

    cos: object
    sin: object
    heads: int
    half: int


def split_rope_freqs(grid: np.ndarray, dim: int, num_heads: int, theta: float,
                     double_precision: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """diffusers ``LTX2AudioVideoRotaryPosEmbed.forward`` / ``LTX2RotaryPosEmbed1d`` (split type).

    ``grid``: ``[T, P]`` fractional positions (already divided by the max positions).
    Returns cos, sin of shape ``[T, num_heads, dim // num_heads // 2]`` in fp32.
    """
    grid = np.asarray(grid, dtype=np.float32)
    t, p = grid.shape
    steps = dim // (2 * p)
    fdt = np.float64 if double_precision else np.float32
    pow_indices = np.power(fdt(theta), np.linspace(0.0, 1.0, steps, dtype=fdt))
    freqs = (pow_indices * np.pi / 2.0).astype(np.float32)  # [F]
    angles = (grid[:, :, None] * np.float32(2.0) - np.float32(1.0)) * freqs[None, None, :]  # [T, P, F]
    angles = np.transpose(angles, (0, 2, 1)).reshape(t, steps * p).astype(np.float32)  # freq-major
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    pad = dim // 2 - angles.shape[1]
    if pad:
        cos = np.concatenate([np.ones((t, pad), np.float32), cos], axis=1)
        sin = np.concatenate([np.zeros((t, pad), np.float32), sin], axis=1)
    half = dim // num_heads // 2
    return cos.reshape(t, num_heads, half), sin.reshape(t, num_heads, half)


def rope_constants(g: Graph, cos: np.ndarray, sin: np.ndarray) -> RopeTables:
    t, h, r = cos.shape
    return RopeTables(g.const(cos.reshape(1, t, h, 1, r), trt.float32),
                      g.const(sin.reshape(1, t, h, 1, r), trt.float32), h, r)


def gather_rope_rows(g: Graph, rope: RopeTables, rows) -> RopeTables:
    """Rows ``rows`` (int32 [T_local]) of a RoPE table, e.g. one context-parallel shard."""
    return RopeTables(g.gather(rope.cos, rows, 1), g.gather(rope.sin, rows, 1), rope.heads, rope.half)


def apply_split_rope(g: Graph, x, rope: RopeTables):
    """``x``: ``[B, T, H*2r]`` -> same shape/dtype, rotated in fp32 (``T`` may be a run-time size)."""
    b, t, d = (int(s) for s in x.shape)
    h, r = rope.heads, rope.half
    out_dtype = x.dtype
    x5 = g.reshape(g.cast(x, trt.float32), (b, t, h, 2, r))
    x1 = g.take(x5, 3, 0)
    x2 = g.take(x5, 3, 1)
    o1 = g.sub(g.mul(x1, rope.cos), g.mul(x2, rope.sin))
    o2 = g.add(g.mul(x2, rope.cos), g.mul(x1, rope.sin))
    return g.cast(g.reshape(g.concat([o1, o2], axis=3), (b, t, d)), out_dtype)


def to_heads(g: Graph, x, heads: int):
    """``[B, T, H*D]`` -> ``[B, H, T, D]``."""
    b, t, d = (int(s) for s in x.shape)
    return g.reshape(x, (b, t, heads, d // heads), second=(0, 2, 1, 3))


def from_heads(g: Graph, x):
    """``[B, H, T, D]`` -> ``[B, T, H*D]``."""
    b, h, t, d = (int(s) for s in x.shape)
    return g.reshape(x, (b, t, h * d), first=(0, 2, 1, 3))


class AttnWeights:
    """Weights of one ``LTX2Attention`` (checkpoint layout) fetched lazily from a checkpoint.

    ``linear``: optional ``(g, x, name) -> tensor or None`` provider for the projections (``name`` is
    the checkpoint layer name, e.g. ``transformer_blocks.3.attn1.to_q``). It returns ``None`` for the
    projections it leaves to the bf16 path; the DiT uses it for its NVFP4 projections.
    """

    def __init__(self, ckpt, prefix: str, linear=None):
        self.prefix = prefix
        self.ckpt = ckpt
        self.linear = linear

    def project(self, g: Graph, x, name: str):
        """``x @ W^T + b`` of the projection ``name`` (``to_q``, ``to_out.0``, ...)."""
        return checkpoint_linear(g, self.ckpt, f"{self.prefix}.{name}", x, self.linear)

    def w(self, name: str):
        return self.ckpt.get(f"{self.prefix}.{name}")

    def maybe(self, name: str):
        return self.ckpt.maybe(f"{self.prefix}.{name}")

    def f32(self, name: str):
        return self.ckpt.get(f"{self.prefix}.{name}", np.float32)


def project_qkv(g: Graph, aw: AttnWeights, x_q, x_kv, *, eps: float,
                q_rope: RopeTables | None, k_rope: RopeTables | None, need_q: bool = True):
    """to_q/to_k/to_v + across-heads RMSNorm + split RoPE; returns (q, k, v) as ``[B, T, inner]``."""
    q = None
    if need_q:
        q = aw.project(g, x_q, "to_q")
        q = g.rms_norm(q, aw.f32("norm_q.weight"), eps)
        if q_rope is not None:
            q = apply_split_rope(g, q, q_rope)
    k = aw.project(g, x_kv, "to_k")
    k = g.rms_norm(k, aw.f32("norm_k.weight"), eps)
    if k_rope is not None:
        k = apply_split_rope(g, k, k_rope)
    v = aw.project(g, x_kv, "to_v")
    return q, k, v


def gate_and_project_out(g: Graph, aw: AttnWeights, ctx, x_q, heads: int, *, gated: bool):
    """Per-head ``2*sigmoid`` gates (from the query input) and ``to_out.0``; ``ctx`` is ``[B, T, inner]``."""
    if gated:
        b, t, inner = (int(s) for s in ctx.shape)
        logits = aw.project(g, x_q, "to_gate_logits")  # [B,T,H]
        gates = g.mul(g.sigmoid(g.cast(logits, trt.float32)), g.scalar(2.0, trt.float32, 3))
        gates = g.cast(g.reshape(gates, (b, t, heads, 1)), ctx.dtype)
        ctx = g.reshape(g.mul(g.reshape(ctx, (b, t, heads, inner // heads)), gates), (b, t, inner))
    return aw.project(g, ctx, "to_out.0")


def ltx_attention(g: Graph, aw: AttnWeights, x_q, x_kv, *, heads: int, eps: float,
                  q_rope: RopeTables | None = None, k_rope: RopeTables | None = None,
                  gated: bool = True, mask=None, stg_keep=None, core=None):
    """Full ``LTX2Attention`` forward (single device).

    ``stg_keep``: optional ``[B, 1, 1]`` mask for spatio-temporal guidance; batch rows
    with 0 replace the attention context by the value projection
    (``torch.lerp(value, attn, mask)`` in ``LTX2PerturbedAttnProcessor``).
    ``core``: optional ``(q, k, v) -> ctx`` over ``[B, H, S, D]`` replacing the bf16 softmax attention
    (the DiT's FP8 attention); not combined with ``mask``.
    """
    q, k, v = project_qkv(g, aw, x_q, x_kv, eps=eps, q_rope=q_rope, k_rope=k_rope)
    q4, k4, v4 = to_heads(g, q, heads), to_heads(g, k, heads), to_heads(g, v, heads)
    if core is not None:
        if mask is not None:
            raise ValueError("a custom attention core does not take a mask")
        ctx = from_heads(g, core(q4, k4, v4))
    else:
        ctx = from_heads(g, g.attention(q4, k4, v4, mask=mask))
    if stg_keep is not None:
        ctx = stg_lerp(g, v, ctx, stg_keep)
    return gate_and_project_out(g, aw, ctx, x_q, heads, gated=gated)


def stg_lerp(g: Graph, value, ctx, keep):
    """Per batch row: ``ctx`` where ``keep`` > 0.5, else ``value`` (STG masks are exactly 0 or 1).

    A select keeps both branches bit-exact, like ``torch.lerp`` at weights 0 and 1.
    """
    rank = len(ctx.shape)
    keep = g.reshape(g.cast(keep, trt.float32), (int(keep.shape[0]),) + (1,) * (rank - 1))
    cond = g.ew(keep, g.scalar(0.5, trt.float32, rank), trt.ElementWiseOperation.GREATER)
    return g.select(cond, ctx, value)


def checkpoint_linear(g: Graph, ckpt, name: str, x, linear=None):
    """The checkpoint linear ``name`` on ``x``: the ``linear`` provider's result, else bf16 ``g.linear``."""
    if linear is not None:
        y = linear(g, x, name)
        if y is not None:
            return y
    return g.linear(x, ckpt.get(f"{name}.weight"), ckpt.maybe(f"{name}.bias"))


def feed_forward(g: Graph, ckpt, prefix: str, x, linear=None):
    """diffusers ``FeedForward(activation_fn="gelu-approximate")``: proj -> GELU(tanh) -> proj."""
    h = checkpoint_linear(g, ckpt, f"{prefix}.net.0.proj", x, linear)
    h = g.gelu_tanh(h)
    return checkpoint_linear(g, ckpt, f"{prefix}.net.2", h, linear)


def modulate(g: Graph, x, scale, shift):
    """``x * (1 + scale) + shift`` in x.dtype (scale/shift broadcast ``[B, 1, D]``)."""
    one = g.scalar(1.0, x.dtype, len(x.shape))
    return g.add(g.mul(x, g.add(one, g.cast(scale, x.dtype))), g.cast(shift, x.dtype))


def connector_rope_tables(seq_len: int, dim: int, heads: int, *, base_seq_len: int, theta: float,
                          double_precision: bool = True):
    """``LTX2RotaryPosEmbed1d`` (split): positions ``arange(L) / base_seq_len``."""
    grid = (np.arange(seq_len, dtype=np.float32) / np.float32(base_seq_len)).reshape(seq_len, 1)
    return split_rope_freqs(grid, dim, heads, theta, double_precision)

