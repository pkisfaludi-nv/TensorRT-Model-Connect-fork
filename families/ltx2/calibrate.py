# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Activation amax calibration of the LTX-2.5 DiT for ``--quantization nvfp4`` / ``--attention fp8``.

Development tool (needs torch + diffusers); the build only reads its JSON output. It runs the diffusers
``LTX2Pipeline`` in bf16 with torch hooks and records, over every denoising step of every prompt:

- the input amax of every ``nn.Linear`` inside ``transformer_blocks`` (``transformer_blocks.3.attn1.to_q``);
- the amax of the attention BMM inputs, Q and K after RMSNorm and RoPE, and V
  (``transformer_blocks.3.attn1.q`` / ``.k`` / ``.v``).

``python -m families.ltx2.calibrate MODEL_DIR --out families/ltx2/calibration/ltx25-distilled-amax.json``

The default calibration set is three prompts that do not include the benchmark (fox) prompt, at
768x448, 121 frames, with all 8 distilled steps.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

PROMPTS = (
    "A busy city street at night in the rain, neon signs reflecting on the wet asphalt, people with umbrellas "
    "walking past, cars driving by with their headlights on.",
    "Close-up of an elderly fisherman mending a net on a wooden pier, waves lapping against the posts, seagulls "
    "calling overhead, warm late-afternoon light.",
    "A drummer playing an energetic solo on a stage, colorful spotlights sweeping across the band, the crowd "
    "cheering and clapping in the background.",
)
SEEDS = (11, 23, 37)


class AmaxRecorder:
    """Context manager recording linear-input and attention-input amax of an ``LTX2VideoTransformer3DModel``."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.amax: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self._handles = []
        self._current: str | None = None
        self._original = None

    def _record(self, name: str, tensor) -> None:
        value = float(tensor.detach().abs().amax().float())
        self.amax[name] = max(self.amax[name], value)
        self.calls[name] += 1

    def __enter__(self) -> "AmaxRecorder":
        import torch
        import diffusers.models.transformers.transformer_ltx2 as tl

        for name, module in self.transformer.named_modules():
            if not name.startswith("transformer_blocks."):
                continue
            if isinstance(module, torch.nn.Linear):
                self._handles.append(module.register_forward_pre_hook(lambda m, a, n=name: self._record(n, a[0])))
            elif isinstance(module, tl.LTX2Attention):
                self._handles.append(module.register_forward_pre_hook(lambda m, a, n=name: self._enter(n)))
                self._handles.append(module.register_forward_hook(lambda m, a, o: self._enter(None)))
        self._original = tl.dispatch_attention_fn

        def dispatch(query, key, value, *args, **kwargs):
            if self._current is not None:
                self._record(f"{self._current}.q", query)
                self._record(f"{self._current}.k", key)
                self._record(f"{self._current}.v", value)
            return self._original(query, key, value, *args, **kwargs)

        tl.dispatch_attention_fn = dispatch
        return self

    def _enter(self, name: str | None) -> None:
        self._current = name

    def __exit__(self, *exc) -> None:
        import diffusers.models.transformers.transformer_ltx2 as tl

        tl.dispatch_attention_fn = self._original
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def result(self) -> dict[str, float]:
        return {k: self.amax[k] for k in sorted(self.amax)}


def main() -> int:
    import torch
    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--frames", type=int, default=121)
    args = ap.parse_args()

    pipe = LTX2Pipeline.from_pretrained(args.model_dir, processor=None, prompt_enhancer=None, duration_head=None,
                                        torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    call = dict(guidance_scale=1.0, audio_guidance_scale=1.0, stg_scale=0.0, audio_stg_scale=0.0,
                modality_scale=1.0, audio_modality_scale=1.0, guidance_rescale=0.0, audio_guidance_rescale=0.0,
                spatio_temporal_guidance_blocks=None, enable_prompt_enhancement=False, use_cross_timestep=True)
    started = time.perf_counter()
    with AmaxRecorder(pipe.transformer) as recorder:
        for prompt, seed in zip(PROMPTS, SEEDS):
            pipe(prompt=prompt, width=args.width, height=args.height, num_frames=args.frames, frame_rate=24.0,
                 sigmas=DISTILLED_SIGMA_VALUES, generator=torch.Generator("cuda").manual_seed(seed),
                 output_type="latent", return_dict=False, **call)
    out = {
        "format": "ltx2-activation-amax-v1",
        "model": "Lightricks/LTX-2.5-Diffusers (distilled transformer)",
        "calibration": {
            "pipeline": "diffusers LTX2Pipeline, bf16, distilled 8-step schedule, guidance off",
            "width": args.width, "height": args.height, "frames": args.frames,
            "prompts": list(PROMPTS), "seeds": list(SEEDS), "steps_per_prompt": len(DISTILLED_SIGMA_VALUES),
            "statistic": "max over all prompts, steps and tokens of |x| (linear inputs; attention Q/K after "
                         "RMSNorm + RoPE, V)",
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "seconds": round(time.perf_counter() - started, 1),
            "calls_per_tensor": sorted(set(recorder.calls.values())),
        },
        "amax": recorder.result(),
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["calibration"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
