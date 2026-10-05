# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 NVFP4 / FP8 quantization: offline weight quantization, Q/DQ graph patterns and references.

Two independent, opt-in options of the DiT (``trtmc ltx2 build``):

``--quantization nvfp4`` (GEMMs)
    Weights: E2M1 values packed two per byte (even element in the low nibble), one E4M3 scale per
    16 input channels and one fp32 global scale per tensor (``amax / (6 * 448)``), computed here
    from the bf16 checkpoint at build time. Activations: TensorRT ``IDynamicQuantizeLayer`` computes
    the per-16 block scales at run time; only its fp32 global scale (``amax / (6 * 448)``) is a
    calibrated constant (``calibration/``). The GEMM is ``DQ(dyn_q(x)) @ DQ(w)^T`` in a strongly typed
    network, which TensorRT lowers to an FP4 tensor-core GEMM.

``--attention fp8`` (video self-attention)
    Q, K and V are quantized to E4M3 with calibrated per-tensor scales (Q and K carry
    ``sqrt(1/sqrt(d))`` each, as in the ModelOpt FP8 SDPA export), and the softmax output is
    quantized to E4M3 with scale 1/448 inside a fused (non-decomposable) ``IAttention``. The
    attention output stays bf16.

The ``fake_*`` NumPy functions are the references the parity tests compare the engines against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

from .cli import ATTENTION_CHOICES, QUANTIZATION_CHOICES

NVFP4_BLOCK = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
SOFTMAX_FP8_SCALE = 1.0 / FP8_MAX
_FP4 = ml_dtypes.float4_e2m1fn
_FP8 = ml_dtypes.float8_e4m3fn


@dataclass(frozen=True)
class NvFp4Weight:
    packed: np.ndarray  # uint8 [N, K/2]: E2M1 codes, element 2i in the low nibble
    block_scales: np.ndarray  # uint8 [N, K/16]: E4M3 bytes
    global_scale: float
    out_features: int
    in_features: int


def _fp8_bytes(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, np.float32).astype(_FP8).view(np.uint8))


def _fp8_values(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw, np.uint8).view(_FP8).astype(np.float32)


def _fp4_round(values: np.ndarray) -> np.ndarray:
    """Round to E2M1 (nearest even, saturating at +-6)."""
    return np.clip(values, -FP4_MAX, FP4_MAX).astype(_FP4)


def nvfp4_global_scale(amax: float) -> float:
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError(f"NVFP4 needs a positive finite amax, got {amax}")
    return float(np.float32(amax) / np.float32(FP4_MAX * FP8_MAX))


def _nvfp4_blocks(x: np.ndarray, global_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """E4M3 block scales ``[.., K/16]`` (as fp32 values) and E2M1 values ``[.., K]`` of ``x``."""
    x = np.asarray(x, np.float32)
    k = x.shape[-1]
    if k % NVFP4_BLOCK:
        raise ValueError(f"NVFP4 needs the input width ({k}) divisible by {NVFP4_BLOCK}")
    blocks = x.reshape(*x.shape[:-1], k // NVFP4_BLOCK, NVFP4_BLOCK)
    gs = np.float32(global_scale)
    amax = np.abs(blocks).max(axis=-1)
    scales = (amax / np.float32(FP4_MAX) / gs).astype(_FP8).astype(np.float32)
    denom = (scales * gs)[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(denom > 0, blocks / np.where(denom > 0, denom, 1), 0).astype(np.float32)
    return scales, _fp4_round(q).astype(np.float32).reshape(x.shape)


def quantize_nvfp4_weight(weight: np.ndarray) -> NvFp4Weight:
    """NVFP4 weight from a ``[out, in]`` checkpoint tensor (global scale from its own amax)."""
    w = np.asarray(weight, np.float32)
    n, k = (int(s) for s in w.shape)
    gs = nvfp4_global_scale(float(np.abs(w).max()))
    scales, values = _nvfp4_blocks(w, gs)
    codes = values.astype(_FP4).view(np.uint8) & np.uint8(0x0F)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << np.uint8(4))).astype(np.uint8)
    return NvFp4Weight(packed=np.ascontiguousarray(packed), block_scales=_fp8_bytes(scales), global_scale=gs,
                       out_features=n, in_features=k)


def dequantize_nvfp4_weight(w: NvFp4Weight) -> np.ndarray:
    """fp32 ``[out, in]`` values the engine multiplies with (reference)."""
    lo = (w.packed & np.uint8(0x0F)).view(_FP4).astype(np.float32)
    hi = (w.packed >> np.uint8(4)).view(_FP4).astype(np.float32)
    values = np.empty((w.out_features, w.in_features), np.float32)
    values[:, 0::2], values[:, 1::2] = lo, hi
    scales = _fp8_values(w.block_scales) * np.float32(w.global_scale)
    return (values.reshape(w.out_features, -1, NVFP4_BLOCK) * scales[..., None]).reshape(values.shape)


def fake_quant_nvfp4(x: np.ndarray, global_scale: float) -> np.ndarray:
    """Reference of the dynamic NVFP4 activation quantize + dequantize (blocks along the last axis)."""
    scales, values = _nvfp4_blocks(x, global_scale)
    shape = values.shape
    v = values.reshape(*shape[:-1], -1, NVFP4_BLOCK)
    return (v * (scales * np.float32(global_scale))[..., None]).reshape(shape)


def fp8_scale(amax: float) -> float:
    if not np.isfinite(amax) or amax <= 0:
        raise ValueError(f"FP8 needs a positive finite amax, got {amax}")
    return float(np.float32(amax) / np.float32(FP8_MAX))


def fake_quant_fp8(x: np.ndarray, scale: float) -> np.ndarray:
    """Reference of a per-tensor E4M3 quantize + dequantize (saturating)."""
    s = np.float32(scale)
    q = np.clip(np.asarray(x, np.float32) / s, -FP8_MAX, FP8_MAX).astype(_FP8).astype(np.float32)
    return q * s


# ---------------------------------------------------------------------- graph patterns


def nvfp4_linear(g, x, w: NvFp4Weight, act_global_scale: float, bias: np.ndarray | None = None):
    """``x @ W^T + b`` as an NVFP4 GEMM: ``x`` is ``[B, T, K]`` bf16 (``T`` may be a run-time size)."""
    import tensorrt as trt

    b, t, k = (int(s) for s in x.shape)
    if k != w.in_features:
        raise ValueError(f"NVFP4 linear: input width {k} != weight width {w.in_features}")
    n = w.out_features
    out_dtype = x.dtype
    x2 = g.reshape(x, (-1, k))
    g._keepalive.extend((w.packed, w.block_scales))
    w_fp4 = g.net.add_constant((n, k), trt.Weights(trt.DataType.FP4, w.packed.ctypes.data, n * k)).get_output(0)
    w_bs = g.net.add_constant((n, k // NVFP4_BLOCK), trt.Weights(trt.DataType.FP8, w.block_scales.ctypes.data,
                                                                  w.block_scales.size)).get_output(0)
    w_scale = g.net.add_dequantize(w_bs, g.const(np.float32([w.global_scale]), trt.float32), out_dtype).get_output(0)
    w_dq = g.net.add_dequantize(w_fp4, w_scale, out_dtype)
    w_dq.axis = 1
    a_gs = g.const(np.float32([act_global_scale]), trt.float32)
    dyn = g.net.add_dynamic_quantize(x2, 1, NVFP4_BLOCK, trt.DataType.FP4, trt.DataType.FP8)
    dyn.set_input(1, a_gs)
    a_scale = g.net.add_dequantize(dyn.get_output(1), a_gs, out_dtype).get_output(0)
    a_dq = g.net.add_dequantize(dyn.get_output(0), a_scale, out_dtype)
    a_dq.axis = 1
    y = g.net.add_matrix_multiply(a_dq.get_output(0), trt.MatrixOperation.NONE, w_dq.get_output(0),
                                  trt.MatrixOperation.TRANSPOSE).get_output(0)
    y = g.reshape(g.cast(y, out_dtype), (b, t, n))
    if bias is not None:
        y = g.add(y, g.const(bias, out_dtype, shape=(1, 1, n)))
    return y


def _fp8_qdq(g, x, scale: float):
    import tensorrt as trt

    s = g.const(np.float32(scale), trt.float32, shape=())
    q = g.net.add_quantize(x, s, trt.DataType.FP8).get_output(0)
    return g.net.add_dequantize(q, s, x.dtype).get_output(0)


@dataclass(frozen=True)
class AttentionScales:
    """Per-tensor E4M3 scales of one attention's BMM inputs (Q and K include ``sqrt(1/sqrt(d))``)."""

    q: float
    k: float
    v: float
    softmax: float = SOFTMAX_FP8_SCALE

    @staticmethod
    def from_amax(q_amax: float, k_amax: float, v_amax: float, head_dim: int) -> "AttentionScales":
        root = float(np.sqrt(1.0 / np.sqrt(head_dim)))
        return AttentionScales(q=fp8_scale(q_amax * root), k=fp8_scale(k_amax * root), v=fp8_scale(v_amax))


def fp8_attention(g, q, k, v, scales: AttentionScales):
    """Softmax attention over ``[B, H, S, D]`` bf16 with FP8 BMM inputs and an FP8 softmax (bf16 output)."""
    import tensorrt as trt

    root = float(np.sqrt(1.0 / np.sqrt(int(q.shape[-1]))))
    q = _fp8_qdq(g, g.mul(q, g.scalar(root, q.dtype, 4)), scales.q)
    k = _fp8_qdq(g, g.mul(k, g.scalar(root, k.dtype, 4)), scales.k)
    v = _fp8_qdq(g, v, scales.v)
    layer = g.net.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    if layer is None:
        raise RuntimeError("TensorRT rejected the FP8 attention layer")
    layer.normalization_quantize_scale = g.const(np.float32(scales.softmax), trt.float32, shape=())
    layer.normalization_quantize_to_type = trt.DataType.FP8
    layer.decomposable = False
    return layer.get_output(0)


def fake_fp8_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scales: AttentionScales) -> np.ndarray:
    """Reference of :func:`fp8_attention` (``[B, H, S, D]`` fp32 arrays).

    Like a fused FP8 attention kernel, the unnormalized softmax numerator ``exp(s - rowmax)`` (at most 1)
    is quantized for the second GEMM, and the fp32 row sum normalizes the result afterwards.
    """
    d = q.shape[-1]
    root = np.float32(np.sqrt(1.0 / np.sqrt(d)))
    qq = fake_quant_fp8(q * root, scales.q)
    kq = fake_quant_fp8(k * root, scales.k)
    vq = fake_quant_fp8(v, scales.v)
    s = qq @ np.swapaxes(kq, -1, -2)
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    return (fake_quant_fp8(p, scales.softmax) @ vq) / p.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------- calibration data


@dataclass(frozen=True)
class DitQuantization:
    """What the DiT builder quantizes and with which calibrated activation statistics.

    ``amax`` maps checkpoint layer names (``transformer_blocks.3.attn1.to_q``) to the input amax
    of that linear, and ``transformer_blocks.3.attn1.{q,k,v}`` to the attention BMM input amax
    (after Q/K RMSNorm and RoPE).
    """

    gemm: str = "none"
    attention: str = "bf16"
    amax: dict | None = None
    weight_cache: dict | None = None

    def __post_init__(self) -> None:
        if self.gemm not in QUANTIZATION_CHOICES:
            raise ValueError(f"ltx2 quantization must be one of {QUANTIZATION_CHOICES}, got {self.gemm!r}")
        if self.attention not in ATTENTION_CHOICES:
            raise ValueError(f"ltx2 attention must be one of {ATTENTION_CHOICES}, got {self.attention!r}")
        if self.enabled and not self.amax:
            raise ValueError("ltx2 NVFP4 / FP8 builds need calibrated activation amax values")

    @property
    def enabled(self) -> bool:
        return self.gemm != "none" or self.attention != "bf16"

    def input_amax(self, name: str) -> float:
        value = (self.amax or {}).get(name)
        if value is None:
            raise KeyError(f"no calibrated activation amax for {name}")
        return float(value)

    def attention_scales(self, prefix: str, head_dim: int) -> AttentionScales:
        return AttentionScales.from_amax(self.input_amax(f"{prefix}.q"), self.input_amax(f"{prefix}.k"),
                                         self.input_amax(f"{prefix}.v"), head_dim)


CALIBRATION_DIR = Path(__file__).with_name("calibration")
DEFAULT_CALIBRATION = CALIBRATION_DIR / "ltx25-distilled-amax.json"


def load_amax(path: Path = DEFAULT_CALIBRATION) -> dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    amax = data["amax"]
    for name, value in amax.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"invalid calibrated amax for {name}: {value!r}")
    return {str(k): float(v) for k, v in amax.items()}
