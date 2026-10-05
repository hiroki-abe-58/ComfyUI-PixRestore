# Verification (v0.1.0)

Facts below are from runs on 2026-10-05. Numbers are in [results/verification_summary.json](results/verification_summary.json)
and [results/demo_metrics.json](results/demo_metrics.json).

## What "official" means here

- **Code:** `inference.py` of [csslc/PixRestore](https://github.com/csslc/PixRestore) at
  `909ca06e614cf72e218d52b4fa41077c603d872b`, unmodified, run as a single process (`python inference.py ...`).
  The README's 8-process `accelerate launch --multi_gpu` example was not used.
- **Model:** `pixrestore-s` from [VCLab-PolyU/PixRestore](https://huggingface.co/VCLab-PolyU/PixRestore) at
  `a5fe719a0c517422cff9f7a4895ab303e2320d51`: `clean_weights/ema_model.safetensors` and the `config.json` next to it
  (inference.py's default). `configs/train.yaml` was not used. The released config is the one-step DINO-GAN model
  (`fixed_train_t: 1.0`, `use_dino_gan: true`, `cfg_ratio: 0.0`, `mixed_precision: bf16`, degradation token on).
- **DINOv2:** ViT-S/14 from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) at
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (a local clone) and `dinov2_vits14_pretrain.pth`, passed with
  `--dinov2-repository` / `--dinov2-checkpoint`. The released config's own paths (`/dinov2`, ...) belong to the authors' machine.
- **Arguments:** `--infer-steps 1 --cfg-scale 1.0 --seed 0 --test-mode center_crop`. One step means exactly one call of the
  denoiser (checked with a forward hook in every run). Image *i* of an input folder is seeded with *i*.
- **Environment:** upstream pins `torch==2.4.0`, which has no kernels for the RTX 5090 (SM120). The reference environment
  uses torch 2.14.1+cu130 and otherwise the upstream pins (accelerate 0.34.2, numpy 1.26.4, timm 1.0.30).

The upstream DiT has `@torch.compile` on `LightningDiTBlock.forward`, `FinalLayer.forward`, `SwiGLUFFN.forward`,
`modulate` and `modulate_adasin`. On Windows this needs Triton: without it the unmodified run stops with
`TritonMissing` at the first compiled call. So there are two official modes:

| mode | how | result |
|---|---|---|
| eager | `TORCHDYNAMO_DISABLE=1` (torch.compile returns the undecorated functions) | runs; **the node reproduces this mode** |
| as written (compiled) | `triton-windows 3.8.0.post29` added to a copy of the reference environment | runs; differs from eager, see below |

Compiled vs eager on the 12 demo inputs (same weights, inputs and seeds): the DINOv2 input, all six DINOv2 feature
layers and the noise are identical; the DiT output differs. 8-bit output: 11.3 % of values differ, at most 5/255,
mean 0.11/255. Final float in [0, 1]: at most 0.020. The cause is the inductor-compiled kernels under bf16 autocast;
both modes are deterministic (two compiled runs and three eager runs were each identical to themselves).

## Results with the released weights (GPU)

Windows 11 native, RTX 5090 32 GB (driver 595.95), torch 2.14.1+cu130, ComfyUI v0.38.0, Python 3.12.13.

| check | data | result |
|---|---|---|
| official eager: plain CLI vs three runs that record intermediate tensors | 12 demo inputs | PNGs byte-identical (12/12); all 144 recorded tensors identical across runs |
| node runtime vs official eager, per layer | 12 | input in [-1, 1], noise, 6 DINOv2 layers, model output (float32), final float, 8-bit image: all identical (12/12) |
| ComfyUI HTTP queue vs official eager PNG | 12 (exact), 2 (center crop on 512x512), 2 (center crop of 768x576 and 400x300) | identical pixels and identical layer hashes |
| same prompt twice, A then B then A | ComfyUI started with `--cache-none` | executed each time (run ids 17, 18, 19, 20), identical outputs |
| Unload, then the same prompt | | model loaded again (load id 2), identical output |
| non-512 input in exact mode | 768x576, 768x768 | refused with a message naming the size and the center-crop option; next prompt OK |
| interrupt during a 64-image batch | | stopped after 18 images (`execution_interrupted`); next prompt OK and identical |
| global flags | ComfyUI sets `allow_fp16_bf16_reduction_math_sdp(True)` | official defaults set during the call, ComfyUI's values back afterwards; CUDA RNG state restored |
| clean install | `git archive` of the tested commit into `custom_nodes/pixrestore-clean-install` of a fresh ComfyUI v0.38.0 with a fresh venv, `tools/setup_pixrestore.py` downloading all 28 files | all of the above repeated: same results |
| frontend | the two example workflows loaded with the ComfyUI frontend's `loadApiJson` | no missing nodes, `graphToPrompt` round trip identical |

Measured on that machine (one process, sequential prompts; not a benchmark):
first prompt including import, hash check and load 3.7 s; then about 0.11 s per prompt (median, 11 prompts; GPU part
about 0.045 s); peak CUDA memory allocated by the node 342 MiB. The ComfyUI process with the model loaded used at
most 5.3 GiB private memory.

## CPU tests (CI, Ubuntu and Windows)

No released weights: random weights with the released architecture (`tests/make_cpu_reference.py`). The unmodified
official `inference.py` runs on the CPU in its own environment (upstream pins, `TORCHDYNAMO_DISABLE=1`, bf16 autocast);
the node runtime must reproduce its PNGs exactly in a ComfyUI environment. Also covered: pinned file hashes, refusal of
changed or missing upstream files, the official center crop (compared with upstream's own function), the 8-bit
round trip through ComfyUI's LoadImage/SaveImage conversions for all 256 values, flag restore, config checks, and that
the comparison fails when the seed changes or autocast is removed.

## Not verified

- Other GPUs, Linux or WSL GPU runs, real weights on the CPU, Apple MPS.
- The compiled (Triton) mode inside ComfyUI; the node always runs eager.
- PixRestore B / L / XL, multi-step sampling, CFG > 1, other resolutions as model input, tiling.
- Image quality in general: the demo is 4 synthetic images x 3 synthetic degradations, measured with PSNR only.
- Non-512 sources are center-cropped exactly like the official `--test-mode center_crop`; the quality of that path on
  real photographs was not evaluated.
