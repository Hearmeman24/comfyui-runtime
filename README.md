# comfyui-runtime

Shared runtime for the HearmemanAI ComfyUI RunPod pod templates (wan, minimax, ltx2, qwen-image):
the download managers, the model provisioner, the shared boot script, the model validator, the base
image Dockerfile, and the SageAttention wheel build tooling. The small CivitAI downloader is a
checksummed, provenance-recorded runtime snapshot so `stable` controls its rollout and rollback.

Templates do not vendor these files. Each template clones this repo at boot at the `runtime_ref`
pinned in its own `pins.json` (alongside `base_image`). All four currently pin the **branch**
`stable`, so a merge to `main` deploys nowhere on its own: the runtime goes live when `stable`
moves, and then it reaches every template at once on the next pod boot.

**Start with [`ARCHITECTURE.md`](./ARCHITECTURE.md)** — how the templates and this runtime fit
together, the three tiers of change and their blast radius, the promotion model, and what
`template.json` is for. Read it before `CONTRACTS.md`.

This repo is public because the templates must clone it unauthenticated at boot. A private repo
would force a PAT into every public template.

The repo must stay blob-free: no wheels, no model weights, no binaries. Every pod clones this repo
on every boot, so repo size is boot time. SageAttention wheels ship as GitHub Release assets and
are fetched by checksum, never committed.

`CONTRACTS.md` is the frozen interface: file layouts, env vars, exit codes, and the decisions
behind them. Read it before changing anything.

## Layout

| Path | What it is |
|---|---|
| `CONTRACTS.md` | Frozen interfaces and decisions |
| `src/hf_download_manager.py` | Unified model downloader (HF, aria2c, gdown) |
| `src/civitai_downloads.sh` | Validates, prepares, schedules, and joins runtime-owned CivitAI downloads |
| `vendor/civitai_downloader/` | Canonical CivitAI downloader snapshot, dependency pin, checksum, license notice, and provenance |
| `src/provisioner.py` | Registry walker, workflow copier, quant rewrite |
| `src/start.sh` | Shared boot script |
| `src/sage_probe.py` | SageAttention kernel probe |
| `tools/validate_models.py` | Registry vs workflow validator (CI gate) |
| `tools/build_sage_wheel.sh` + `tools/BUILD_WHEELS.md` | Wheel build on the H200 pod |
| `tools/test_*.py` | Stdlib self-test suites, run by CI `verify` |
| `torch/cu128.txt` `torch/cu130.txt` | The family torch pins |
| `base/Dockerfile` | The shared base image, two CUDA variants |

## Approved ComfyUI base

The base pins the latest stable ComfyUI release v0.42.1 to commit
`63be2f9b82ad0dac88a7ceaa28a9cf3154125a18`. It keeps native FLUX.3 Klein support (BFL's fixed
Klein workflows need v0.41.3 or newer) and Qwen Image 2.1 support; the build checks for both
model implementations. Torch and SageAttention pins are unchanged.

Release this base before merging the dependent template PRs:

1. Merge the runtime PR and create the next `vN` release tag through the existing release process.
2. Wait for CircleCI to publish both immutable base images:
   `hearmeman/comfyui-base:cu128-comfy0.42.1-63be2f9b-torch2.11.0` and
   `hearmeman/comfyui-base:cu130-comfy0.42.1-63be2f9b-torch2.11.0`.
3. Merge the template PRs (qwen, ltx2, minimax, wan) that update their Dockerfile and `pins.json`
   to the new cu130 base, then release a new image per template. Existing pods keep their
   image's approved core until redeployed.

Moving `stable` alone does not upgrade an image's baked ComfyUI checkout. Previous immutable base
tags (`comfy0.41.0-98cc0bf8`, `comfy0.36.0-3dd559d8`) remain available for rollback.

## License

AGPL-3.0, inherited from ComfyUI. See `LICENSE`.

The vendored CivitAI downloader retains its upstream MIT declaration and provenance under
`vendor/civitai_downloader/`.
