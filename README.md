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
| `versions.json` + `tools/versions.py` | Single source of truth for the ComfyUI pin, torch, CUDA variants and the template list |
| `tools/comfyui_release_watch.py` + `.github/workflows/comfyui-release-watch.yml` | Daily check that opens the runtime bump PR |
| `tools/fanout_templates.py` | Opens one template PR per new base, skipping held templates |

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

## Upgrade flow

1. **Watch.** `ComfyUI release watch` (GitHub Actions, daily 08:17 Israel time, or run it manually)
   compares the latest stable ComfyUI release with `versions.json`. If upstream is newer, it rewrites
   `versions.json` and the `base/Dockerfile` ARG defaults (`tools/versions.py set-release`) and
   opens `auto/comfyui-<label>` against `main`.
2. **Test build.** On any non-`main` branch that changes base inputs (`base/`, `torch/`,
   `versions.json`, `tools/versions.py`, CircleCI config), `test_build_base` builds cu130 exactly as
   a release would, with no login and no push. `verify` runs `tools/versions.py check`, so a
   Dockerfile default or torch pin that drifts from `versions.json` fails the PR.
3. **Release.** Merge, then push the next `vN` tag. `build_base` derives every tag and build arg from
   `versions.json` and pushes both variants.
4. **Fan-out.** `fanout_templates` then opens `auto/comfyui-<label>` in every template listed in
   `versions.json` that is behind and not held, moving `pins.json` `base_image` and the Dockerfile
   `ARG BASE_IMAGE`. Held templates are skipped and named in the job log. It never merges, tags or
   touches RunPod; each template still ships through its own `vN` tag.

To bump by hand: `python3 tools/versions.py set-release v0.43.0 <40-char sha>`.

### Holding a template back

A template stays on its current base image and `runtime_ref` while the others move on when its
`pins.json` has a `hold` with a reason:

```json
{
  "runtime_ref": "v4",
  "base_image": "hearmeman/comfyui-base:cu130-comfy0.42.1-63be2f9b-torch2.11.0",
  "hold": {"reason": "AI-Toolkit breaks on ComfyUI v0.43.0", "since": "2026-10-20"}
}
```

The override lives in the template's own `pins.json`, not in a central matrix, because that file
already owns the template's `base_image` and `runtime_ref`. The hold sits next to the values it
freezes, it is reviewed in that template's history, and no second list can drift from it. Pinning
`runtime_ref` to a tag in the same file keeps a held template off `stable` promotions. Delete `hold`
to rejoin the next fan-out (re-run it with `tools/fanout_templates.py`, or wait for the next base).
`tools/fanout_templates.py --dry-run` prints the plan without writing anything.

Setup the fan-out needs: a CircleCI context `template-fanout` holding `TEMPLATE_FANOUT_GITHUB_TOKEN`,
a fine-grained token with Contents and Pull requests write on the four template repos. Without it
the job prints a dry run and passes.

## License

AGPL-3.0, inherited from ComfyUI. See `LICENSE`.

The vendored CivitAI downloader retains its upstream MIT declaration and provenance under
`vendor/civitai_downloader/`.
