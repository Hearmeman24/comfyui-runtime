# comfyui-runtime

Shared runtime for the HearmemanAI ComfyUI RunPod pod templates (wan, minimax, ltx2, qwen-image):
the download manager, the model provisioner, the shared boot script, the model validator, the base
image Dockerfile, and the SageAttention wheel build tooling.

Templates do not vendor these files. Each template clones this repo at boot and pins it by commit
SHA in its own `pins.json` (`runtime_ref` plus `base_image`). A push here deploys nowhere on its
own; a template picks up a change only when its pin is bumped.

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
| `src/provisioner.py` | Registry walker, workflow copier, quant rewrite |
| `src/start.sh` | Shared boot script |
| `src/sage_probe.py` | SageAttention kernel probe |
| `tools/validate_models.py` | Registry vs workflow validator (CI gate) |
| `tools/build_sage_wheel.sh` + `tools/BUILD_WHEELS.md` | Wheel build on the H200 pod |
| `tools/test_*.py` | Stdlib self-test suites, run by CI `verify` |
| `torch/cu128.txt` `torch/cu130.txt` | The family torch pins |
| `base/Dockerfile` | The shared base image, two CUDA variants |

## License

AGPL-3.0, inherited from ComfyUI. See `LICENSE`.
