# BUILD_WHEELS.md - the SageAttention wheel runbook

How to build, gate, publish and consume the two universal SageAttention
wheels. Written to be re-run cold months from now: follow it top to bottom,
do not improvise steps. Source of truth for the design is
`docs/specs/2026-08-12-shared-runtime-scaffold/plan.md` section 5 and
`CONTRACTS.md` section 10.

## What this produces

Two wheels, one per CUDA major, each covering every GPU arch SageAttention
can dispatch (sm80, sm89, sm90, sm120, sm121; sm86 needs no cubin, it uses
the triton JIT path):

| Build | Variant | Torch trio | Published as (GitHub Release tag on `Hearmeman24/comfyui-runtime`) |
|---|---|---|---|
| A | cu128 | `torch/cu128.txt` | `sage-d1a57a5-cu128-torch2.11.0` |
| B | cu130 | `torch/cu130.txt` | `sage-d1a57a5-cu130-torch2.11.0` |

The tag grammar is `sage-<SAGE_COMMIT short>-<cuNNN>-torch<version>`. The two
wheels share a filename, so each lives in its own Release tag.

The wheel is a one-off durable artifact (plan D7). It is consumed unchanged
by `base/Dockerfile` (both wheels `ADD`ed into every base image) and never
built per template, per deployment, or at boot.

## When to re-run

Re-run the affected variant(s) whenever any of these change:

- a line in `torch/cu128.txt` or `torch/cu130.txt` (the canonical trios)
- `SAGE_COMMIT` in `tools/build_sage_wheel.sh` (currently `d1a57a546`)
- `TORCH_CUDA_ARCH_LIST` in the same script (currently `8.0 8.9 9.0 12.0 12.1`)

Publishing a new Release deploys nothing by itself: wheels are baked, so
every existing image keeps its matched pair. The rollout is plan.md 5c
"Sequencing": wheels, then new base tags, then per-template `pins.json`
bumps.

## Hard rules

1. **Every command on the pod goes through the `runpod-ssh` MCP**
   (`exec` / `upload` / `download` / `list_pods`). Never `ssh` or `scp` from
   Bash. A pasted SSH connection string identifies the pod; it does not
   authorize shelling out. If the MCP is down, stop and ask Aviv to restart
   it.
2. **Pod lifecycle (create / terminate) goes through the RunPod REST API**
   (`https://rest.runpod.io/v1`, header `Authorization: Bearer $RUNPOD_API_KEY`).
3. **Every pod creation is a paid action and needs an explicit go from Aviv
   first**, with the cost estimate shown. Approval never carries forward
   between runs.
4. **The run ends with terminate-and-confirm** (step 10), success or failure.
   A forgotten H200 is the real cost risk of this whole procedure: at
   $4.59/hr it burns about $110/day doing nothing.

## Cost and capacity (verified 2026-08-12)

- H200 SXM on secure cloud is **$4.59/hr** and its stockStatus currently
  reads **"Low"**. Build B needs a CUDA 13.0 host (R580 driver), which
  narrows the host pool further, so **build B may have to wait for
  capacity**. Re-check price and stock at run time; do not assume these
  numbers held.
- Expected compile time is 30 to 90 minutes per build (single-arch boot
  builds run 3 to 5 minutes today; this builds five arch targets). Budget
  $10 to $15 for both builds. These are estimates, not measurements.

## Pod spec per build

No network volume for either: everything happens on local NVMe under `/tmp`
and the only artifact leaving the pod is one ~28 MB wheel.

| | Build A (cu128) | Build B (cu130) |
|---|---|---|
| GPU | 1x H200 SXM, secure cloud | 1x H200 SXM, secure cloud |
| CUDA version filter | 12.8 | 13.0 or newer (needs an R580 host) |
| Container image | the current `hearmeman/comfyui-wan-template:vN` (check Docker Hub / the RunPod template for the newest `vN`) | the current `hearmeman/comfyui-minimax-template:vN` (same check) |
| Container disk | 100 GB (the image is 20 to 30 GB; leave compile headroom) | 100 GB |
| Network volume | none | none |

Why these images: both are CUDA `-devel` bases (nvcc and cuobjdump present)
with the venv at `/opt/venv`. The build script installs the canonical trio
itself and asserts it, so the image's own torch state does not matter beyond
being the right CUDA major family. Once the shared base image exists, its
matching variant tag is an equally good (and smaller) pod image.

## Procedure, per build

Run build A first, then B. Steps 2 to 10 repeat per variant.

### 1. Preflight, local

- Confirm `torch/cu128.txt` / `torch/cu130.txt` say what you intend to build
  against. The trio in that file IS the wheel's ABI contract.
- Confirm `src/sage_probe.py` exists (it ships with the runtime repo; the
  gate needs it on the pod).
- Get Aviv's explicit go, showing the pod spec above and the cost estimate.

### 2. Create the pod (RunPod REST API)

Find the GPU type id first (do not guess it):

```
GET https://rest.runpod.io/v1/gputypes
```

Pick the H200 SXM entry; note its `id`, current price and stockStatus. Then
create the pod. Payload sketch (field names drift; verify against the
current REST docs before sending):

```
POST https://rest.runpod.io/v1/pods
{
  "name": "sage-wheel-<variant>",
  "imageName": "<container image from the table>",
  "gpuTypeIds": ["<H200 SXM id>"],
  "gpuCount": 1,
  "cloudType": "SECURE",
  "containerDiskInGb": 100,
  "allowedCudaVersions": ["12.8"]        // build B: ["13.0", "13.1", ...]
}
```

If creation fails with no capacity (expected for build B while stock is
Low), stop and report; do not retry in a loop at $4.59/hr risk elsewhere.

### 3. Wait for the pod, confirm via the MCP

`runpod-ssh list_pods` until the new pod shows up running. Record its pod id
and name; every following step targets it.

### 4. Upload the three inputs

Via `runpod-ssh upload`, into `/tmp/sagebuild/` on the pod:

- `tools/build_sage_wheel.sh`
- `torch/<variant>.txt` (i.e. `cu128.txt` or `cu130.txt`, kept under that
  exact filename; the script resolves `<script dir>/<variant>.txt`)
- `src/sage_probe.py`

### 5. Run the build, detached

The compile runs 30 to 90 minutes, longer than any sane exec timeout, so
start it detached and poll the log:

```
runpod-ssh exec: bash -c 'cd /tmp/sagebuild && nohup bash build_sage_wheel.sh cu128 > build.log 2>&1 & echo started'
```

(`cu130` for build B.) Poll with:

```
runpod-ssh exec: tail -n 40 /tmp/sagebuild/build.log
```

The script is `set -e` end to end. Outcomes:

- **Success**: the log ends with `== wheel built and gated (<variant>) ==`,
  the wheel path, and a `sha256sum` line. Both gates have passed ON the pod:
  the cuobjdump cubin assertions (`_qattn_sm80` has sm_80; `_qattn_sm89` has
  sm_89, sm_120 and sm_121; `_qattn_sm90` has sm_90a) and a real kernel
  launch of the freshly installed wheel on the H200.
- **Failure**: an `ERROR:` line (or a compiler error) near the end of the
  log. See Known risk below for the one failure mode we half expect. Either
  way, continue to step 10 and terminate the pod.

### 6. Record the sha256

Copy the `sha256sum` line from the log verbatim. It goes into the Release
notes and the Dockerfile checksum.

### 7. Download the wheel

Get the exact filename first:

```
runpod-ssh exec: ls /tmp/sage_build/wheel/
```

then `runpod-ssh download` it to a durable local path, for example
`~/src/comfy/comfyui-runtime/dist/<variant>/` (untracked; wheels never enter
git, the repo must stay blob-free so the boot clone stays fast).

### 8. Verify the local copy

`shasum -a 256` (macOS) on the downloaded file. It must equal the on-pod
value from step 6 exactly. If it does not, the download is corrupt; delete
and re-download. Never publish a wheel whose local hash you have not
matched against the on-pod hash.

### 9. (Build A done) Repeat steps 2 to 10 for build B

### 10. Terminate and confirm - ALWAYS, success or failure

```
DELETE https://rest.runpod.io/v1/pods/{podId}
```

Then confirm with `runpod-ssh list_pods` that the pod is GONE from the
listing. Do not end the session between create and this confirmation. If
the listing still shows it, keep terminating until it is gone and say so.

### 11. Publish the Release (outward-facing: explicit go from Aviv first)

One Release per variant on `Hearmeman24/comfyui-runtime`:

```
gh release create sage-d1a57a5-cu128-torch2.11.0 \
    dist/cu128/sageattention-*.whl \
    --repo Hearmeman24/comfyui-runtime \
    --title "SageAttention d1a57a546, cu128, torch 2.11.0" \
    --notes "<see below>"
```

The notes MUST record, per plan.md section 5 step 4:

- the wheel's sha256 (from step 6/8)
- the full torch trio it links (paste `torch/<variant>.txt`'s three pins)
- `SAGE_COMMIT=d1a57a546`
- `TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 12.0 12.1"`
- the build pod GPU (H200 SXM) and date

Release tags are production infrastructure once `base/Dockerfile` points at
them: never delete or retag a published sage Release.

### 12. Fill the Dockerfile placeholders

`base/Dockerfile` carries two deliberately unshippable `ADD` placeholders
(FIXME URLs plus all-zero checksums; the base image cannot build until this
step). For each variant, replace:

- the URL, with the real Release asset URL (take the filename verbatim from
  the Release; do not trust the guessed one)
- the `--checksum=sha256:...` value, with the wheel's real sha256

Then the base build's variant-matched import assertion is the standing drift
gate: a trio bump without a wheel re-run dies at image build time, never on
a customer pod.

## Known risk, documented not solved

`-gencode` flags are shared across ALL extensions in SageAttention's
`setup.py`, so the sm90a source must also compile under 8.0, 8.9, 12.0 and
12.1. The first pod run is the proof. If it fails, the fallback is a small
per-extension gencode patch kept IN THIS REPO (`comfyui-runtime`) and
applied by `build_sage_wheel.sh` after the `reset --hard`, never a fork of
SageAttention. A failed compile costs one pod-hour, not a release.

## Troubleshooting

- **"venv does not match the canonical trio"**: the pip install of the trio
  file did not land (network, index outage) or the trio file uploaded is
  stale. Fix the input; never hand-edit versions on the pod to make the
  assertion pass.
- **"build pod GPU is smXX, not sm90"**: wrong pod. The kernel-launch gate
  only proves the sm90a cubins if it runs on an sm90 card. Terminate and
  recreate with the H200 spec.
- **cuobjdump assertion failure**: the wheel is missing an arch. Check
  `SAGE_COMMIT` (at the old `68de379` pin, `setup.py` ignores
  `TORCH_CUDA_ARCH_LIST` and builds only for the visible GPU) and that
  `TORCH_CUDA_ARCH_LIST` reached the build (the script exports it).
- **Probe failure after both installs**: the wheel imports but cannot launch
  a kernel. Do not publish. This is the exact failure class this whole
  procedure exists to keep off customer pods.
- **No CUDA 13 capacity**: expected while H200 stock is Low. Build A does
  not depend on build B; publish A and wait for B rather than substituting a
  different GPU.
