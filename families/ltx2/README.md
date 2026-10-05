# LTX-2.5 (`ltx2`)

Text-to-audio-video for Lightricks LTX-2.5 diffusers checkpoints (`LTX2Pipeline`, for example
`Lightricks/LTX-2.5-Diffusers`). One bundle generates a video and its 48 kHz stereo soundtrack.

## Scope

- Task `text_to_audio_video`, precision `bf16`, batch 1. Opt-in NVFP4 DiT GEMMs and FP8 video
  self-attention (see [NVFP4 GEMMs and FP8 attention](#nvfp4-gemms-and-fp8-attention)).
- The distilled transformer with its 8-step schedule (guidance 1, no STG). The bundle bakes
  the schedule into `runtime.json`. The full transformer (30 steps, CFG/STG/modality
  guidance) is not supported yet.
- The video size and frame count are fixed at build time (`--image-height`, `--image-width`,
  `--video-num-frames`). Height and width must be multiples of 32, and the frame count must
  be `8n+1`. The default is 960x544, 121 frames at 24 fps.
- `--context-parallel-size 1` runs on one GPU. `--context-parallel-size 2` splits the
  video tokens of the DiT across two GPUs. The text encoder runs on every rank. The video
  VAE decodes in tiles that the ranks share, and the last rank decodes the audio
  (see [Tiled video decode](#tiled-video-decode)).

## Bundle

| Section | Contents |
|---|---|
| `text_encoder.plan` | Gemma 4 text tower and the LTX-2 text connectors (video and audio context) |
| `denoiser.plan` | Joint audio/video DiT. With CP=2, one plan serves both ranks. Two-stage bundles serve the full and the half-resolution grid. |
| `vae.plan` | Video VAE decoder for one tile shape (whole video with `--vae-tile-pixels 0 --vae-tile-frames 0`) |
| `audio.plan` | Audio VAE decoder and vocoder with bandwidth extension |
| `latent_upsampler.plan` | Two-stage bundles only: the 2x spatial latent upsampler |
| `tokenizer.json`, `runtime.json` | Tokenizer, shapes and schedule |

Context parallelism keeps the audio stream and text replicated and shards the video
tokens. Video self-attention all-gathers each rank's normed and rotated keys and values.
Video-to-audio attention merges per-rank softmax statistics through one small all-gather.
The network uses no all-to-all collective.

## Tiled video decode

The video VAE decodes the latent video as overlapping tiles of one shape, so one static plan serves
every tile. The tiles are blended with linear ramps over their overlaps and normalized by the summed
weights, as in the Lightricks and TensorRT-LLM `tiled_decode`. Tiles are at most 512 pixels with at
least 64 pixels of overlap; clips longer than 257 frames also split in time, into tiles of up to
256 frames that overlap by at least 24 frames. `families/ltx2/vae_tiling.py` computes the tile plan
and writes it into `runtime.json`.

- With context parallelism, the ranks decode disjoint tiles. The worker ranks send their decoded
  tiles to rank 0 over NCCL point-to-point on the engines' communicator, and rank 0 blends every
  tile in tile order. The blended video is the same bit for bit as the single-GPU tiled decode.
- The last rank decodes the audio while rank 0 decodes its tiles. On one GPU, the host blend runs
  while the GPU decodes the audio.
- A transfer that does not finish within 10 minutes aborts the NCCL communicator instead of
  leaving NCCL kernels running.

`trtmc ltx2 build` (`python -m tensorrt_model_connect ltx2 build`) takes the tile options
`--vae-tile-pixels`, `--vae-tile-overlap-pixels`, `--vae-tile-frames` and
`--vae-tile-overlap-frames`. A size of 0 leaves that axis untiled, and setting both sizes to 0
builds the untiled decoder. The shared `trtmc build --family ltx2` uses the defaults.

## Two-stage pipeline

`trtmc ltx2 build --two-stage` builds a bundle that also runs the two-stage recipe of the diffusers
LTX-2.5 documentation (distilled checkpoint in both stages, no LoRA). Run it with
`--set two_stage=true`; single-stage generation stays the default, also on two-stage bundles.

1. Stage 1 denoises at half the width and height with the 8 distilled sigmas.
2. The latent upsampler (`latent_upsampler/`) doubles the latent grid.
3. The video and audio latents are re-noised to the first stage 2 sigma
   (`noise_scale * noise + (1 - noise_scale) * latents`, noise_scale 0.909375). The noise draws
   continue the seeded stream after the stage 1 noise.
4. Stage 2 refines at full resolution with `STAGE_2_DISTILLED_SIGMA_VALUES`
   (0.909375, 0.725, 0.421875), then the video and audio are decoded.

Width and height must be multiples of 64, and the checkpoint must include `latent_upsampler/`. One
DiT plan serves both grids: the video token count is a run-time dimension, and the RoPE tables of
both grids are constants that the plan selects by token count. Stage 1 runs about a quarter of the
full-resolution tokens, so the two-stage run does 8 small steps and 3 full steps instead of 8 full
steps. Context parallelism applies to both stages, and every rank upsamples the latents itself.

## NVFP4 GEMMs and FP8 attention

Two opt-in DiT options of `trtmc ltx2 build`. The default stays bf16. The text encoder, the VAEs, the
upsampler and the vocoder are never quantized.

- `--quantization nvfp4` runs the linears on the video token stream as NVFP4 GEMMs. These are the
  self-attention Q/K/V/out, the text cross-attention Q/out, the audio-to-video Q/out, the
  video-to-audio K/V and the feed-forward. They carry the DiT's GEMM cost at full resolution.
  - The audio stream (251 tokens), the text-side K/V, the gate logits, AdaLN, `proj_in` and
    `proj_out` stay bf16. They are a small share of the FLOPs and the most sensitive to
    quantization.
  - Weights are quantized at build time from the bf16 checkpoint. The E2M1 values get one E4M3
    scale per 16 input channels and one fp32 global scale per tensor.
  - Activations are quantized at run time by `IDynamicQuantizeLayer` (per-16 E4M3 block scales).
    Only its fp32 global scale is calibrated: the calibrated amax with 2x headroom, divided by
    6 * 448.
- `--attention fp8` runs the video self-attention with E4M3 Q, K and V (calibrated per-tensor
  scales, with the softmax scale folded into Q and K) and an E4M3 softmax, in one fused
  `IAttention`. The attention output stays bf16.

TensorRT lowers both patterns to FP4 and FP8 tensor-core kernels. The family emits only the Q/DQ
graph.

### Calibration

`calibration/ltx25-distilled-amax.json` holds the input amax of every DiT block linear and of every
attention's Q, K and V (after RMSNorm and RoPE). `calibrate.py` (`python -m families.ltx2.calibrate
MODEL_DIR --out ...`, needs torch and diffusers) records them with torch hooks on the diffusers
`LTX2Pipeline` in bf16.

- Calibration set: three prompts (a rainy city street, a fisherman on a pier, a drummer on stage)
  with seeds 11, 23 and 37, at 768x448 and 121 frames, over all 8 distilled steps.
- The benchmark prompt is not part of the set.
- `--calibration PATH` selects another file. `runtime.json` records the options and the SHA-256 of
  the calibration file under `dit_quantization`.

### Measurements

Measured on RTX PRO 6000 Blackwell with TensorRT-RTX 1.7.1: 1280x704, 241 frames, seed 42, default
clocks, medians of 3.

| Configuration | 1 GPU | 2 GPUs (CP=2) |
|---|---|---|
| bf16, single-stage, untiled VAE | 47.0 s (5.39 s/step) | 23.5 s (2.54 s/step) |
| NVFP4 only | 30.6 s (3.41 s/step) | 19.6 s (2.04 s/step) |
| FP8 attention only | 41.3 s (4.70 s/step) | 21.0 s (2.23 s/step) |
| NVFP4 + FP8 attention | 25.3 s (2.76 s/step) | 17.2 s (1.74 s/step) |
| bf16, two-stage, tiled VAE | 25.5 s | 13.7 s |
| NVFP4 + FP8 attention, two-stage, tiled VAE | 15.8 s | 11.4 s |

At equal locked clocks (1792 MHz), NVFP4 + FP8 attention is 1.72x faster than bf16 on one GPU and
1.37x faster with CP=2 (single-stage).

Context parallelism gains less on a quantized DiT. The fp16 K/V all-gather costs about 7.5 ms per
layer over PCIe (about 0.36 s per step at 27k tokens) and does not shrink with the GEMMs and the
attention.

Quantization changes the denoising trajectory, and NVFP4 accounts for most of the change.

- FP8 attention alone stays close to bf16: about 21 dB PSNR against the bf16 output of the same
  seed.
- NVFP4 moves the output like a different run of the same scene: about 17-19 dB, against 28.7 dB
  between bf16 on 1 and 2 GPUs. Its 1-GPU and 2-GPU outputs also differ more from each other.
- Depending on the trajectory, fur detail is softer and fast motion shows more ghosting.

Use NVFP4 as a speed option, not as a bf16-equivalent output.

## Build and run

```bash
python -m tensorrt_model_connect build /models/LTX-2.5-Diffusers --family ltx2 \
  --backend trt_rtx --precision bf16 \
  --image-width 960 --image-height 544 --video-num-frames 121 \
  --context-parallel-size 2 -o ltx25-cp2.bundle

trtmc generate-video ltx25-cp2.bundle --runtime-root <trtmc runtime root> \
  --prompt "A red fox walking through a snowy forest at dawn" \
  --output out --set seed=42
```

`generate-video` writes `out/frame-NNNNNN.png` and `out/audio.wav`. Set `TRTMC_LTX2_PROGRESS=1`
to print one progress line per stage and denoising step.

## TensorRT-RTX

- Context parallelism with `--backend trt_rtx` requires TensorRT-RTX 1.7.1 or newer.
  TensorRT-RTX 1.6.x has no multi-device support, so the build stops with an error
  for `--context-parallel-size 2` when an older version is installed.
- Build and run with the same TensorRT-RTX version. A bundle's plans are specific to the
  version that built them.
- TRTMC does not pin `tensorrt-rtx` in `pyproject.toml`. Install the Python package that
  matches your TensorRT-RTX runtime.

## Native Windows with two GPUs

- Put both GPUs in TCC mode. Under WDDM or MCDM they report no peer access and NCCL has
  no transport.
- Build `nccl.dll` from source and point the runtime at it with `TRTMC_NCCL_LIBRARY`.
  See [Multi-Device Execution](../../website/docs/features/multi-device.md).
- Start the ranks with `tools/launch_ranks.py`, because native Windows has no `mpirun`:

```powershell
python tools\launch_ranks.py -n 2 --gpus 0,1 --nccl-library C:\nccl\install\bin\nccl.dll `
  --library-dir <TensorRT-RTX bin directory> `
  -- trtmc generate-video ltx25-cp2.bundle --runtime-root <trtmc runtime root> `
     --prompt "..." --output out --set seed=42
```

To give each rank its own TensorRT-RTX runtime cache, put `{rank}` in the path, for example
`--runtime-cache cache-rank{rank}.bin`.

## Tests

- `tests/test_*_parity.py` build each engine from tiny random weights and compare it
  with diffusers, including the latent upsampler and the two-grid DiT plan.
- `tests/test_context_parallel.py` runs the CP=2 DiT on two GPUs (torch-free ranks) against
  the single-device plan and diffusers. It also runs the tile-parallel VAE decode and checks that
  it matches the single-GPU tiled decode bit for bit. It needs `TRTMC_NCCL_LIBRARY`.
- `tests/test_quantization.py` checks the NVFP4 weight packing and the fake-quant references. On a
  Blackwell GPU, it also compares one NVFP4 linear and one FP8 attention engine with those references,
  and the tiny DiT (NVFP4, FP8 attention, both) with diffusers under the same fake quantization.
  The CP=2 test also runs the quantized plan on two ranks.
- `tests/test_vae_tiling.py` checks the tile plan (coverage, ramps, rank assignment) without a GPU.
- `tests/test_e2e.py` builds the real checkpoint and runs the native CLI. It compares the
  output with `LTX2Pipeline` started from the same noise. Select it with `--e2e-model ltx2`.

### E2E thresholds

Both sides run the 48-layer DiT in bf16. A single forward of the native engine differs from
an fp32 diffusers forward by about 3% relative L2, as does diffusers in bf16. The 8 distilled
steps amplify that difference into detail-level changes of the same scene. Exact pixel
agreement is therefore not expected. The thresholds separate "same trajectory" from
"different or broken output". They were set from these measurements on 2x RTX PRO 6000 with
TensorRT-RTX 1.7.1:

| Comparison (seed 42) | Frame PSNR mean / min | Log-spectrogram corr | RMS ratio |
|---|---|---|---|
| Native vs diffusers bf16, 384x640x49 (CP=1) | 19.1 / 18.4 dB | 0.908 | 1.00 |
| Native vs diffusers bf16, 960x544x121 (CP=2) | 18.2 / 16.2 dB | 0.892 | 1.13 |
| Diffusers bf16 vs diffusers fp32 (both sizes) | 22.7-23.8 / 21.9-22.9 dB | 0.92 | 1.00 |
| Diffusers bf16, seed 42 vs seed 7 (unrelated output) | 11.5-12.9 / 10.1-12.2 dB | 0.75-0.81 | 0.25-5.9 |

The thresholds are a mean of at least 15 dB, a minimum of at least 14 dB, a correlation of
at least 0.85 and an RMS ratio in [0.8, 1.25]. Each one sits between the measured native
values and the unrelated-output values. The audio correlation is lower than the video
agreement because the released pipeline runs its vocoder in bf16; against an fp32 vocoder,
the native audio reaches 0.95-0.97.
