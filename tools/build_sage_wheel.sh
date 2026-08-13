#!/usr/bin/env bash
# Build the universal SageAttention wheel for ONE CUDA variant, on a RunPod
# H200 SXM pod (plan.md section 5; CONTRACTS.md section 10). Run it via the
# runpod-ssh MCP following tools/BUILD_WHEELS.md - never by hand-rolled ssh.
#
# Usage:  bash build_sage_wheel.sh <cu128|cu130>
#
# Expects, in the same directory as this script (uploaded together, see the
# runbook):
#   cu128.txt / cu130.txt   the canonical torch trio (a copy of torch/<v>.txt)
#   sage_probe.py           the real-kernel probe (a copy of src/sage_probe.py)
# Overridable via TRIO_FILE / SAGE_PROBE / WORK_DIR env vars.
#
# What it does, in order:
#   1. installs the canonical torch trio and asserts the venv equals it
#      exactly, on an sm90 GPU (exit nonzero if not)
#   2. clones SageAttention and hard-resets to the pinned commit
#   3. pip wheel with TORCH_CUDA_ARCH_LIST covering every dispatch arch
#   4. gates ON THE POD, before anything leaves it:
#      a. cuobjdump every built extension and assert the required cubins
#      b. pip install the fresh wheel and run a real kernel launch
#   5. prints the wheel path and its sha256
#
# Compile invocation lifted from comfyui-minimax/src/sage_build.sh:22,28,81-87.
# The cache-key and volume-cache machinery there is deliberately DROPPED: this
# wheel is a one-off durable artifact published to a GitHub Release (plan D7),
# not a per-pod cache.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

VARIANT=${1:-}
case "$VARIANT" in
    cu128) EXPECTED_CUDA_MAJOR=12 ;;
    cu130) EXPECTED_CUDA_MAJOR=13 ;;
    *) echo "usage: build_sage_wheel.sh <cu128|cu130>" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TRIO_FILE=${TRIO_FILE:-$SCRIPT_DIR/$VARIANT.txt}
PROBE=${SAGE_PROBE:-$SCRIPT_DIR/sage_probe.py}
WORK_DIR=${WORK_DIR:-/tmp/sage_build}

[ -f "$TRIO_FILE" ] || die "torch trio file not found: $TRIO_FILE (upload torch/$VARIANT.txt beside this script)"
[ -f "$PROBE" ] || die "sage_probe.py not found: $PROBE (upload src/sage_probe.py beside this script)"
GENCODE_PATCH=${GENCODE_PATCH:-$SCRIPT_DIR/sage_per_ext_gencode.patch}
[ -f "$GENCODE_PATCH" ] || die "sage_per_ext_gencode.patch not found: $GENCODE_PATCH (upload tools/sage_per_ext_gencode.patch beside this script)"
command -v cuobjdump >/dev/null 2>&1 || die "cuobjdump not on PATH (the pod image must be a CUDA -devel base)"
command -v nvcc >/dev/null 2>&1 || die "nvcc not on PATH (the pod image must be a CUDA -devel base)"

# Pinned build inputs (plan D7). d1a57a546 is REQUIRED, not preferred: at the
# old 68de379 pin, setup.py ignores TORCH_CUDA_ARCH_LIST entirely and an H200
# build yields an sm90-only wheel. 8.6 is deliberately absent from the arch
# list: sm86 routes to the triton JIT path and never touches a cubin
# (upstream core.py:146-147); 12.1 is present on cu130 because core.py:154-155
# dispatches sm121.
export SAGE_COMMIT=d1a57a546
SAGE_REPO=https://github.com/thu-ml/SageAttention.git
# SEMICOLONS ARE LOAD-BEARING. SageAttention's setup.py parses this with
# arch_list_env.replace(",", ";").split(";") and never splits on whitespace
# (upstream setup.py:96 at d1a57a546). A space-separated list therefore
# arrives as ONE item, only capability.startswith("8.0") matches, and the
# build silently produces an sm80-only wheel that passes every import check.
# Verified the hard way on an H200, 2026-08-13: the space-separated form
# built _qattn_sm80 alone and the cuobjdump gate caught it.
#
# THE ARCH LIST IS PER VARIANT (approved 2026-08-13, amends plan decision #8).
# Do not "helpfully" re-add 12.1 to cu128: nvcc 12.8 rejects it outright
# ("nvcc fatal   : Unsupported gpu architecture 'compute_121a'", verified on
# an H200 2026-08-13 - it killed every extension before compiling anything).
# sm121 support landed in CUDA 12.9, so each variant's wheel carries exactly
# what its own nvcc can emit: only cu130 covers sm121. There is no cu128
# workaround: sm_120a cubins are arch-specific and do not run on sm121, and
# the 'f' family suffix needs 12.9+ too.
case "$VARIANT" in
    cu128) export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0" ;;
    cu130) export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0;12.0;12.1" ;;
esac
export EXT_PARALLEL=4
export NVCC_APPEND_FLAGS="--threads 8"
export MAX_JOBS=32

# The trio file is the single source of truth on this pod. The template image
# used as the pod base may carry a /torch-constraint.txt freeze of an OLDER
# resolve (and the future base image exports PIP_CONSTRAINT); it must not pin
# this venv away from the canonical trio we are about to install.
unset PIP_CONSTRAINT

echo "== build_sage_wheel $VARIANT =="
echo "trio file:  $TRIO_FILE"
echo "sage:       $SAGE_COMMIT  arches: $TORCH_CUDA_ARCH_LIST"
nvcc --version | tail -n1

pip install -r "$TRIO_FILE"

# Assert the venv now equals the canonical trio, exactly, and that we are on
# the sm90 build pod the runbook mandates (the kernel probe below only proves
# the sm90a cubins if it actually runs on one). Exit if not.
python3 - "$TRIO_FILE" "$EXPECTED_CUDA_MAJOR" <<'PY'
import re
import sys

want = {}
with open(sys.argv[1]) as fh:
    for line in fh:
        m = re.match(r"^(torch|torchvision|torchaudio)==(\S+)\s*$", line.strip())
        if m:
            want[m.group(1)] = m.group(2)
if sorted(want) != ["torch", "torchaudio", "torchvision"]:
    sys.exit("trio file does not pin exactly torch/torchvision/torchaudio: %s" % sorted(want))

import torch
import torchaudio
import torchvision

have = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
}
bad = ["%s: want %s, have %s" % (k, want[k], have[k])
       for k in want if have[k].split("+")[0] != want[k]]
cuda = torch.version.cuda or "none"
if cuda.split(".")[0] != sys.argv[2]:
    bad.append("torch.version.cuda: want major %s, have %s" % (sys.argv[2], cuda))
cap = torch.cuda.get_device_capability(0)
if cap != (9, 0):
    bad.append("build pod GPU is sm%d%d, not sm90; use an H200 (BUILD_WHEELS.md)" % cap)
if bad:
    sys.exit("venv does not match the canonical build environment:\n  " + "\n  ".join(bad))
print("venv OK: torch %s (cuda %s) on sm%d%d" % (torch.__version__, cuda, cap[0], cap[1]))
PY

SRC_DIR="$WORK_DIR/SageAttention"
WHEEL_DIR="$WORK_DIR/wheel"
UNPACK_DIR="$WORK_DIR/unpack"
# A rerun can leave a previous attempt behind; git refuses to clone into a
# non-empty directory (comfyui-minimax/src/sage_build.sh:77-79).
rm -rf "$SRC_DIR" "$WHEEL_DIR" "$UNPACK_DIR"
mkdir -p "$WORK_DIR"

git clone "$SAGE_REPO" "$SRC_DIR"
git -C "$SRC_DIR" reset --hard "$SAGE_COMMIT"
# Per-extension gencode (the documented fallback, BUILD_WHEELS.md "Known
# risk", proven needed 2026-08-13): upstream shares one -gencode set across
# all extensions, and the Hopper-only sm90 source (wgmma/TMA) dies in ptxas
# when force-compiled for sm_89/sm_120/sm_121 ("Instruction 'wgmma.mma_async
# ...' not supported on .target 'sm_89'"). The patch gives each extension
# only the arches its sources support. Not a fork: applied fresh on every
# build after the reset --hard above.
git -C "$SRC_DIR" apply "$GENCODE_PATCH"

# --no-build-isolation so the extension links against this venv's torch
# (comfyui-minimax/src/sage_build.sh:84-87).
pip wheel --no-deps --no-build-isolation --wheel-dir "$WHEEL_DIR" "$SRC_DIR"

BUILT=$(find "$WHEEL_DIR" -maxdepth 1 -name 'sageattention-*.whl' | head -n1)
[ -n "$BUILT" ] || die "pip wheel produced no sageattention wheel"

# ---- Gate 1: cuobjdump cubin assertions (plan.md section 5, step 3a) -------
# Per-arch dispatch is per module: _qattn_sm80 serves sm80, _qattn_sm89 serves
# sm89 AND sm120/sm121 (fp8 path), _qattn_sm90 serves sm90 via sm90a cubins.
# Any missing arch fails the run before anything leaves the pod. This
# assertion alone would have caught the currently shipped wheel's missing
# sm90 cubins.
python3 -m zipfile -e "$BUILT" "$UNPACK_DIR"

assert_archs() {
    local module=$1; shift
    local so archs want
    so=$(find "$UNPACK_DIR" -name "${module}*.so" | head -n1)
    [ -n "$so" ] || die "wheel contains no ${module}*.so"
    archs=$(cuobjdump "$so" | grep -oE 'sm_[0-9]+a?' | sort -u)
    for want in "$@"; do
        if ! grep -qx "$want" <<<"$archs"; then
            die "$module is missing $want (has: $(tr '\n' ' ' <<<"$archs"))"
        fi
    done
    echo "OK $module: $(tr '\n' ' ' <<<"$archs")"
}

assert_archs _qattn_sm80 sm_80
# The sm89 module's expected cubin set is per variant: cu128 cannot carry
# sm_121a (nvcc 12.8 cannot emit it; see the arch list comment above).
# The Blackwell cubins are the arch-SPECIFIC variants: setup.py maps 12.0 ->
# num="120a" and 12.1 -> num="121a" (code=sm_120a / sm_121a), so cuobjdump
# reports sm_120a / sm_121a. Plain sm_120 / sm_121 never appear in any
# upstream build - asserting them was a gate-spec bug (caught 2026-08-13
# when both otherwise-good wheels failed it).
if [ "$VARIANT" = cu128 ]; then
    assert_archs _qattn_sm89 sm_89 sm_120a
else
    assert_archs _qattn_sm89 sm_89 sm_120a sm_121a
fi
assert_archs _qattn_sm90 sm_90a

# ---- Gate 2: real kernel launch on this H200 (plan.md section 5, step 3b) --
# --force-reinstall because the pod's template image already carries a
# sageattention of this exact version from its baked wheel; without it pip
# calls the requirement satisfied (comfyui-minimax/src/sage_build.sh:53-57).
# sage_probe.py must exit 0; set -e makes anything else fail the run.
pip install --no-deps --force-reinstall "$BUILT"
python3 "$PROBE"

echo
echo "== wheel built and gated ($VARIANT) =="
echo "wheel:  $BUILT"
sha256sum "$BUILT"
