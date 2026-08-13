# BUILD_WHEELS.md - the SageAttention wheel runbook

How to build, gate, publish and consume the two universal SageAttention
wheels. Written to be re-run cold months from now: follow it top to bottom,
do not improvise steps. Source of truth for the design is
`docs/specs/2026-08-12-shared-runtime-scaffold/plan.md` section 5 and
`CONTRACTS.md` section 10.

## What this produces

Two wheels, one per CUDA major, each covering every GPU arch its own nvcc
can emit (sm86 needs no cubin, it uses the triton JIT path):

- cu128: sm80, sm89, sm90, sm120. **Not sm121**: nvcc 12.8 cannot emit
  `compute_121a` (sm121 support landed in CUDA 12.9), verified the hard way
  on an H200 2026-08-13. This costs nothing on RunPod today because no sm121
  card is rentable there.
- cu130: sm80, sm89, sm90, sm120, sm121.

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
- the per-variant `TORCH_CUDA_ARCH_LIST` in the same script (currently
  cu128 `8.0;8.9;9.0;12.0`, cu130 `8.0;8.9;9.0;12.0;12.1`; the semicolons
  are load-bearing, see the comment in the script)
- `tools/sage_per_ext_gencode.patch` (applied by the script after its
  `reset --hard`; see "Per-extension gencode" below)

Publishing a new Release deploys nothing by itself: wheels are baked, so
every existing image keeps its matched pair. The rollout is plan.md 5c
"Sequencing": wheels, then new base tags, then per-template `pins.json`
bumps.

## Hard rules

1. **Every command on the pod goes through the `runpod-ssh` MCP**
   (`exec` / `upload` / `list_pods`). The MCP has **no download tool**; the
   working egress path is `runpodctl send` / `receive` (step 7). Never `ssh`
   or `scp` from Bash. A pasted SSH connection string identifies the pod; it
   does not authorize shelling out. If the MCP is down, stop and ask the maintainer to
   restart it.
2. **Pod lifecycle (create / terminate) goes through the RunPod REST API**
   (`https://rest.runpod.io/v1`, header `Authorization: Bearer $RUNPOD_API_KEY`).
3. **Every pod creation is a paid action and needs an explicit go from the maintainer
   first**, with the cost estimate shown. Approval never carries forward
   between runs.
4. **The run ends with terminate-and-confirm** (step 10), success or failure.
   A forgotten H200 is the real cost risk of this whole procedure: at
   $4.59/hr it burns about $110/day doing nothing.

## Cost and capacity (measured, 2026-08-13 run)

- H200 SXM on secure cloud was **$4.59/hr**, stockStatus read **Medium**,
  and CUDA 13.0 capacity (R580 host, needed for build B) was available
  immediately. Re-check price and stock at run time; do not assume these
  numbers held.
- A successful build's wall clock is **10 to 14 minutes per variant**. The
  whole first run cost **~$8.80 over ~1h55m across three pods** - one cu128
  attempt was terminated and redone, and most of the time went to image
  pulls, debugging and transfer, not compiling. Budget ~$10 for a clean
  re-run of both variants.

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

Run builds A and B **in parallel** (two pods at once): the GPU-hours cost
the same and the wall clock halves. Steps 2 to 10 repeat per variant.

### 1. Preflight, local

- Confirm `torch/cu128.txt` / `torch/cu130.txt` say what you intend to build
  against. The trio in that file IS the wheel's ABI contract.
- Confirm `src/sage_probe.py` exists (it ships with the runtime repo; the
  gate needs it on the pod).
- Get the maintainer's explicit go, showing the pod spec above and the cost estimate.

### 2. Create the pod (RunPod REST API)

There is no `GET /v1/gputypes` endpoint (verified 2026-08-13: it 404s).
`gpuTypeIds` is an enum on `POST /pods`; the H200 SXM id is the literal
string `"NVIDIA H200"`. Current price and stockStatus come from the GraphQL
API, not REST.

```
POST https://rest.runpod.io/v1/pods
{
  "name": "sage-wheel-<variant>",
  "imageName": "<container image from the table>",
  "gpuTypeIds": ["NVIDIA H200"],
  "gpuCount": 1,
  "cloudType": "SECURE",
  "containerDiskInGb": 100,
  "allowedCudaVersions": ["12.8"]        // build B: ["13.0"]
}
```

The `allowedCudaVersions` enum caps at `"13.0"` (verified 2026-08-13:
`"13.1"` is rejected). For build B, `["13.0"]` alone is the whole filter.

If creation fails with no capacity (expected for build B while stock is
Low), stop and report; do not retry in a loop at $4.59/hr risk elsewhere.

### 3. Wait for the pod, confirm via the MCP

`runpod-ssh list_pods` until the new pod shows up running. Record its pod id
and name; every following step targets it.

### 4. Upload the four inputs

Create the remote directory first - `upload` does not create missing parent
directories:

```
runpod-ssh exec: mkdir -p /tmp/sagebuild
```

Then via `runpod-ssh upload`, into `/tmp/sagebuild/` on the pod:

- `tools/build_sage_wheel.sh`
- `tools/sage_per_ext_gencode.patch` (the script dies without it)
- `torch/<variant>.txt` (i.e. `cu128.txt` or `cu130.txt`, kept under that
  exact filename; the script resolves `<script dir>/<variant>.txt`)
- `src/sage_probe.py`

### 5. Run the build, detached

A successful build runs 10 to 14 minutes wall clock (measured 2026-08-13),
still longer than a sane exec timeout, so start it detached and poll the
log:

```
runpod-ssh exec: bash -c 'cd /tmp/sagebuild && nohup bash build_sage_wheel.sh cu128 > build.log 2>&1 & echo started'
```

(`cu130` for build B.) Poll with:

```
runpod-ssh exec: tail -n 40 /tmp/sagebuild/build.log
```

To confirm mid-build that the right arch list actually reached the compile,
do NOT wait for `Target compute capabilities` in the log - pip buffers
`setup.py`'s stdout and only flushes it when the build ends. Instead read
the `-gencode` flags off the live compiler processes:

```
runpod-ssh exec: ps aux | grep -o -- '-gencode[= ]arch=[^ ]*' | sort -u
```

The script is `set -e` end to end. Outcomes:

- **Success**: the log ends with `== wheel built and gated (<variant>) ==`,
  the wheel path, and a `sha256sum` line. Both gates have passed ON the pod:
  the cuobjdump cubin assertions (`_qattn_sm80` has sm_80; `_qattn_sm89` has
  sm_89 plus sm_120a - and sm_121a on cu130 only; `_qattn_sm90` has sm_90a;
  Blackwell cubins are always the arch-specific `a` variants, plain
  sm_120/sm_121 never appear in an upstream build) and a real kernel launch
  of the freshly installed wheel on the H200.
- **Failure**: an `ERROR:` line (or a compiler error) near the end of the
  log. See Known risk below for the one failure mode we half expect. Either
  way, continue to step 10 and terminate the pod.

### 6. Record the sha256

Copy the `sha256sum` line from the log verbatim. It goes into the Release
notes and the Dockerfile checksum.

### 7. Download the wheel

The `runpod-ssh` MCP has no download tool (verified 2026-08-13). The
working egress is `runpodctl send` / `receive`, which needs no config on
the pod. Get the exact filename first:

```
runpod-ssh exec: ls /tmp/sage_build/wheel/
runpod-ssh exec: runpodctl send /tmp/sage_build/wheel/<wheel filename>
```

`send` prints a one-time code. Locally, from the destination directory
(for example `~/src/comfy/comfyui-runtime/dist/<variant>/` - gitignored;
wheels never enter git, the repo must stay blob-free so the boot clone
stays fast):

```
runpodctl receive <code>
```

### 8. Verify the local copy

`shasum -a 256` (macOS) on the downloaded file. It must equal the on-pod
value from step 6 exactly. If it does not, the download is corrupt; delete
and re-download. Never publish a wheel whose local hash you have not
matched against the on-pod hash.

### 9. Same steps for the other variant (run in parallel, see above)

### 10. Terminate and confirm - ALWAYS, success or failure

```
DELETE https://rest.runpod.io/v1/pods/{podId}
```

Then confirm with `runpod-ssh list_pods` that the pod is GONE from the
listing. Do not end the session between create and this confirmation. If
the listing still shows it, keep terminating until it is gone and say so.

### 11. Publish the Release (outward-facing: explicit go from the maintainer first)

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
- the full `SAGE_COMMIT`
- the `TORCH_CUDA_ARCH_LIST` actually used for THAT variant (cu128
  `"8.0;8.9;9.0;12.0"`, cu130 `"8.0;8.9;9.0;12.0;12.1"`)
- the arch coverage proven by the on-pod cuobjdump gate, and that a real
  sm90 kernel launch passed on the build pod
- the build pod GPU (H200 SXM) and date
- for cu128 only: that the wheel does NOT cover sm121, and why (nvcc 12.8
  cannot emit `compute_121a`; sm121 needs CUDA 12.9+)
- that the build applies `tools/sage_per_ext_gencode.patch`, and why

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

## Per-extension gencode (the risk that materialized)

`-gencode` flags are shared across ALL extensions in SageAttention's
`setup.py`, so the Hopper-only sm90 source (wgmma/TMA) gets force-compiled
for 8.0, 8.9, 12.0 and 12.1 too - and ptxas rejects it ("Instruction
'wgmma.mma_async ...' not supported on .target 'sm_89'"). Proven on the
first pod run, 2026-08-13. The documented fallback is now the standing
mechanism: `tools/sage_per_ext_gencode.patch` gives each extension only the
arches its sources support, and `build_sage_wheel.sh` applies it fresh on
every build after the `reset --hard`. Never fork SageAttention; if the
patch stops applying after a `SAGE_COMMIT` bump, rebase the patch here.

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
- **No CUDA 13 capacity**: did not happen on the 2026-08-13 run (capacity
  was immediate), but if it does: build A does not depend on build B;
  publish A and wait for B rather than substituting a different GPU.
