# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 GEMM / FP8 attention of the LTX-2.5 DiT (``--quantization nvfp4``, ``--attention fp8``).

- CPU: the NVFP4 weight packing, the NumPy fake-quant references and the calibration file.
- GPU: one NVFP4 linear and one FP8 attention engine against the NumPy fake-quant references, and the
  tiny-random DiT built with both options against diffusers with the same fake quantization applied
  to the same layers (and against plain fp32 diffusers, as a sanity bound).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from families.ltx2 import quantization as Q


def test_nvfp4_codes_and_packing() -> None:
    w = np.array([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]],
                 np.float32)
    q = Q.quantize_nvfp4_weight(w)
    # amax 6 -> global scale 6 / 2688, block scale 448 (exact in E4M3): the values are the E2M1 grid.
    assert q.global_scale == pytest.approx(6.0 / (6.0 * 448.0))
    assert q.packed.shape == (1, 8) and q.block_scales.shape == (1, 1)
    assert q.packed[0, 0] == 0x10  # element 0 (code 0) in the low nibble, element 1 (0.5 -> code 1) high
    assert q.packed[0, 3] == 0x76  # 4.0 (code 6) low, 6.0 (code 7) high
    assert q.packed[0, 4] == 0xA9  # -0.5 (code 9) low, -1.0 (code 10) high
    np.testing.assert_array_equal(Q.dequantize_nvfp4_weight(q), w)


def test_nvfp4_weight_matches_the_activation_reference() -> None:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((48, 64)) * 0.02).astype(np.float32)
    w[3, 5] = 0.4  # one outlier block
    q = Q.quantize_nvfp4_weight(w)
    dq = Q.dequantize_nvfp4_weight(q)
    np.testing.assert_array_equal(dq, Q.fake_quant_nvfp4(w, q.global_scale))
    err = np.linalg.norm(dq - w) / np.linalg.norm(w)
    assert 0.02 < err < 0.15  # 4-bit values with per-16 scales: a few percent, never exact


def test_fake_quant_fp8_saturates_and_rounds() -> None:
    x = np.array([0.0, 1.0, 1.0625, 1.1875, 1000.0, -1000.0], np.float32)
    out = Q.fake_quant_fp8(x, 1.0)
    np.testing.assert_array_equal(out, [0.0, 1.0, 1.0, 1.25, 448.0, -448.0])


def test_attention_scales_fold_the_softmax_scale_into_q_and_k() -> None:
    s = Q.AttentionScales.from_amax(8.0, 4.0, 2.0, head_dim=64)
    root = np.sqrt(1.0 / 8.0)
    assert s.q == pytest.approx(8.0 * root / 448.0)
    assert s.k == pytest.approx(4.0 * root / 448.0)
    assert s.v == pytest.approx(2.0 / 448.0)
    assert s.softmax == pytest.approx(1.0 / 448.0)


def test_dit_quantization_options_need_calibration() -> None:
    assert not Q.DitQuantization().enabled
    with pytest.raises(ValueError):
        Q.DitQuantization(gemm="fp8", amax={"x": 1.0})
    with pytest.raises(ValueError):
        Q.DitQuantization(attention="fp4", amax={"x": 1.0})
    with pytest.raises(ValueError):
        Q.DitQuantization(gemm="nvfp4")
    quant = Q.DitQuantization(gemm="nvfp4", attention="fp8", amax={"a.q": 2.0, "a.k": 2.0, "a.v": 1.0})
    with pytest.raises(KeyError):
        quant.input_amax("b.to_q")
    assert quant.attention_scales("a", 128).v == pytest.approx(1.0 / 448.0)


def test_shipped_calibration_covers_every_quantized_tensor_of_the_checkpoint() -> None:
    from families.ltx2.dit_builder import NVFP4_VIDEO_LINEARS

    data = json.loads(Q.DEFAULT_CALIBRATION.read_text(encoding="utf-8"))
    amax = Q.load_amax(Q.DEFAULT_CALIBRATION)
    layers = 48  # LTX-2.5 transformer/config.json num_layers
    for i in range(layers):
        for name in NVFP4_VIDEO_LINEARS:
            assert f"transformer_blocks.{i}.{name}" in amax
        for part in "qkv":
            assert f"transformer_blocks.{i}.attn1.{part}" in amax
    assert data["calibration"]["steps_per_prompt"] == 8
    assert len(data["calibration"]["prompts"]) >= 3
    assert not any("fox" in p.lower() for p in data["calibration"]["prompts"])  # benchmark prompt held out


def test_load_amax_rejects_invalid_values(tmp_path) -> None:
    path = tmp_path / "amax.json"
    path.write_text(json.dumps({"amax": {"a": 0.0}}), encoding="utf-8")
    with pytest.raises(ValueError):
        Q.load_amax(path)


# ---------------------------------------------------------------------- engines (GPU)


def _gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for LTX-2.5 engine tests")
    major, _ = torch.cuda.get_device_capability(0)
    if major < 10:
        pytest.skip("NVFP4 / FP8 tensor cores need a Blackwell GPU (sm_100+)")
    return torch


def _build(fn):
    import tensorrt as trt

    from families.ltx2.graph import Graph, build_plan, make_logger, new_network

    builder, net = new_network(make_logger())
    g = Graph(net)
    fn(g, net, trt)
    return build_plan(builder, net, label="quantization test")


def test_nvfp4_linear_matches_fake_quant_reference() -> None:
    torch = _gpu()
    from families.ltx2.tests.engine_runner import cosine, rel_l2, run_plan

    rng = np.random.default_rng(1)
    s, k, n = 77, 256, 96  # odd token count; K a multiple of 16
    x = rng.standard_normal((1, s, k)).astype(np.float32)
    x[..., ::37] *= 6.0
    import ml_dtypes

    x = x.astype(ml_dtypes.bfloat16).astype(np.float32)  # the engine input is bf16
    w = (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)
    bias = (0.1 * rng.standard_normal(n)).astype(np.float32)
    wq = Q.quantize_nvfp4_weight(w)
    gs = Q.nvfp4_global_scale(float(np.abs(x).max()) * 2.0)

    def fn(g, net, trt):
        xi = net.add_input("x", trt.bfloat16, (1, s, k))
        g.mark_output(Q.nvfp4_linear(g, xi, wq, gs, bias), "y", trt.float32)

    got = run_plan(_build(fn), {"x": torch.from_numpy(x)})["y"].cpu()
    ref = torch.from_numpy(Q.fake_quant_nvfp4(x, gs) @ Q.dequantize_nvfp4_weight(wq).T + bias)
    exact = torch.from_numpy(x @ w.T + bias)
    print(f"nvfp4 linear: vs fake-quant cos {cosine(got, ref):.6f} relL2 {rel_l2(got, ref):.3e} | "
          f"vs unquantized relL2 {rel_l2(got, exact):.3e}")
    # Same quantized operands: only bf16 output rounding and accumulation order differ.
    assert rel_l2(got, ref) < 1e-2
    # And it is really quantized (an unquantized GEMM would be ~1e-3 from exact).
    assert rel_l2(got, exact) > 3e-2


def test_fp8_attention_matches_fake_quant_reference() -> None:
    torch = _gpu()
    from families.ltx2.tests.engine_runner import cosine, rel_l2, run_plan

    rng = np.random.default_rng(2)
    b, h, s, d = 1, 4, 200, 64
    q, k, v = (rng.standard_normal((b, h, s, d)).astype(np.float32) * f for f in (1.5, 1.5, 1.0))
    scales = Q.AttentionScales.from_amax(float(np.abs(q).max()), float(np.abs(k).max()), float(np.abs(v).max()), d)

    def fn(g, net, trt):
        ins = [net.add_input(name, trt.bfloat16, (b, h, s, d)) for name in "qkv"]
        g.mark_output(Q.fp8_attention(g, *ins, scales), "y", trt.float32)

    feed = {name: torch.from_numpy(a) for name, a in zip("qkv", (q, k, v))}
    got = run_plan(_build(fn), feed)["y"].cpu()
    ref = torch.from_numpy(Q.fake_fp8_attention(q, k, v, scales))
    sc = q @ np.swapaxes(k, -1, -2) / np.sqrt(d)
    p = np.exp(sc - sc.max(-1, keepdims=True))
    exact = torch.from_numpy((p / p.sum(-1, keepdims=True)) @ v)
    budget = rel_l2(ref, exact)  # the error FP8 Q/K/V/softmax rounding introduces on this data
    print(f"fp8 attention: vs fake-quant cos {cosine(got, ref):.6f} relL2 {rel_l2(got, ref):.3e} | "
          f"vs exact relL2 {rel_l2(got, exact):.3e} | fake-quant vs exact relL2 {budget:.3e}")
    # The fused kernel rounds in its own order (it does not reproduce the reference bit for bit; on
    # sm_120 about 60% of the FP8 error is shared), so the checks are: it quantizes like the reference
    # (closer to it than to exact attention), and its error has the size FP8 rounding explains.
    assert rel_l2(got, ref) < rel_l2(got, exact)
    assert 0.25 * budget < rel_l2(got, exact) < 1.5 * budget


# ---------------------------------------------------------------------- tiny DiT (GPU)


def _fake_quant_reference(torch, model, quant, shape, inp, tp):
    """diffusers fp32 forward with the engine's quantization applied to the same layers (NumPy references)."""
    import diffusers.models.transformers.transformer_ltx2 as tl

    from families.ltx2.dit_builder import NVFP4_AMAX_HEADROOM, NVFP4_VIDEO_LINEARS

    model = model.to("cuda", torch.float32)
    saved = {}
    handles = []
    for name, module in model.named_modules():
        parts = name.split(".", 2)
        if (quant.gemm == "nvfp4" and len(parts) == 3 and parts[0] == "transformer_blocks"
                and parts[2] in NVFP4_VIDEO_LINEARS):
            w = module.weight.detach().to(torch.bfloat16).float().cpu().numpy()
            saved[name] = module.weight.detach().clone()
            dq = Q.dequantize_nvfp4_weight(Q.quantize_nvfp4_weight(w))
            module.weight.data.copy_(torch.from_numpy(dq))
            gs = Q.nvfp4_global_scale(quant.input_amax(name) * NVFP4_AMAX_HEADROOM)

            def pre(m, a, gs=gs):
                x = a[0].detach().to(torch.bfloat16).float().cpu().numpy()
                return (torch.from_numpy(Q.fake_quant_nvfp4(x, gs)).to(a[0].device, a[0].dtype),) + tuple(a[1:])
            handles.append(module.register_forward_pre_hook(pre))
    current = {"name": None}
    for name, module in model.named_modules():
        if name.endswith(".attn1") and isinstance(module, tl.LTX2Attention):
            handles.append(module.register_forward_pre_hook(lambda m, a, n=name: current.update(name=n)))
            handles.append(module.register_forward_hook(lambda m, a, o: current.update(name=None)))
    original = tl.dispatch_attention_fn

    def dispatch(query, key, value, *args, **kwargs):
        if current["name"] is None or quant.attention != "fp8":
            return original(query, key, value, *args, **kwargs)
        sc = quant.attention_scales(current["name"], int(query.shape[-1]))
        # diffusers layout [B, S, H, D]; the reference works on [B, H, S, D]
        arr = [t.detach().to(torch.bfloat16).float().cpu().numpy().transpose(0, 2, 1, 3) for t in (query, key, value)]
        out = Q.fake_fp8_attention(*arr, sc).transpose(0, 2, 1, 3)
        return torch.from_numpy(np.ascontiguousarray(out)).to(query.device, query.dtype)

    tl.dispatch_attention_fn = dispatch
    try:
        return tp._reference(model, shape, inp, torch.float32)
    finally:
        tl.dispatch_attention_fn = original
        for h in handles:
            h.remove()
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in saved:
                    module.weight.copy_(saved[name])


@pytest.mark.parametrize("gemm,attention", [("nvfp4", "bf16"), ("none", "fp8"), ("nvfp4", "fp8")])
def test_quantized_dit_tiny_parity(tmp_path, gemm, attention) -> None:
    torch = _gpu()
    from families.ltx2.calibrate import AmaxRecorder
    from families.ltx2.dit_builder import build_dit_engine
    from families.ltx2.tests import test_dit_parity as tp
    from families.ltx2.tests.engine_runner import cosine, run_plan

    model, folder, shape = tp.tiny_checkpoint(tmp_path)
    inp = tp._inputs(shape)
    with AmaxRecorder(model) as recorder:  # calibrate on the same inputs (fp32 forward)
        tp._reference(model, shape, inp, torch.float32)
    quant = Q.DitQuantization(gemm=gemm, attention=attention, amax=recorder.result())
    plan = build_dit_engine(folder, shape, stg_blocks=(tp.STG_BLOCK,), quant=quant)
    keep = {"stg_keep": torch.ones(shape.batch), "av_keep": torch.ones(shape.batch)}
    got = run_plan(plan, {**inp, **keep})
    fv, fa = _fake_quant_reference(torch, model, quant, shape, inp, tp)
    rv, ra = tp._reference(model, shape, inp, torch.float32)
    gv, ga = got["video_velocity"].cpu(), got["audio_velocity"].cpu()
    cv, ca = cosine(gv, fv), cosine(ga, fa)
    uv, ua = cosine(gv, rv), cosine(ga, ra)
    print(f"tiny DiT gemm={gemm} attention={attention}: vs fake-quant diffusers video cos {cv:.6f} audio cos "
          f"{ca:.6f} | vs fp32 diffusers video cos {uv:.6f} audio cos {ua:.6f} | fake-quant vs fp32 video cos "
          f"{cosine(fv, rv):.6f}")
    # The engine implements the same quantized network as the fake-quant reference ...
    assert cv > 0.995 and ca > 0.995
    # ... and stays close to the unquantized model.
    assert uv > 0.97 and ua > 0.97
