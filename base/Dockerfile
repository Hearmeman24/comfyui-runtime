# syntax=docker/dockerfile:1
# ============================================================================
# hearmeman/comfyui-base - the shared base image for every HearmemanAI ComfyUI
# pod template (plan.md section 5c, CONTRACTS.md section 6). Templates build
# FROM this and add only their node set, baked weights and start_script.sh.
#
# Two variants from this one file, selected by build args:
#
#   cu128 (default):
#     docker build -f base/Dockerfile -t hearmeman/comfyui-base:cu128-... .
#   cu130:
#     docker build -f base/Dockerfile \
#       --build-arg CUDA_BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04 \
#       --build-arg CUDA_VARIANT=cu130 \
#       -t hearmeman/comfyui-base:cu130-... .
#
# The build CONTEXT is the comfyui-runtime repo ROOT (torch/${CUDA_VARIANT}.txt
# is COPYed in), so always build with `-f base/Dockerfile` from the repo root.
#
# Tag grammar (plan.md 5c): hearmeman/comfyui-base:<cuNNN>-comfy<ref>-torch<ver>
# e.g. cu128-comfy0.34.0-torch2.11.0. A torch or ComfyUI bump means a new tag;
# old tags are immutable rollbacks.
#
# A CUDA_VARIANT / CUDA_BASE_IMAGE mismatch is caught by the sage assertion
# below (torch.version.cuda major must match the variant), and an unknown
# CUDA_VARIANT fails immediately at the trio COPY (no such torch/ file).
# ============================================================================
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG CUDA_VARIANT=cu128

FROM ${CUDA_BASE_IMAGE} AS runtime-common
ARG CUDA_VARIANT

# ComfyUI pin. Must be a branch or tag name (the clone below uses --branch).
# The base tag carries this ref, and COMFYUI_VERSION=approved restores to it
# at boot via /comfyui-approved-ref (plan.md 5b).
ARG COMFYUI_REF=v0.34.0

# Consolidated environment variables (comfyui-wan/Dockerfile:22-26, byte-similar
# in all four templates).
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    HF_XET_HIGH_PERFORMANCE=1

# apt + python3.12 + the venv. Union of comfyui-wan/Dockerfile:28-42 and qwen's
# extra libgles2/libegl1 (comfyui-qwen-image/Dockerfile:20); qwen's deadsnakes
# 3.11 block dies with plan D6 (python 3.12 family-wide).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        curl ffmpeg ninja-build git aria2 git-lfs wget vim \
        libgl1 libgles2 libegl1 libglib2.0-0 build-essential gcc && \
    \
    # make Python3.12 the default python & pip
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    \
    python3.12 -m venv /opt/venv && \
    \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# The canonical torch trio, installed FROM the variant's torch/cuNNN.txt
# (plan D10). That file is the single source of truth: it carries the
# --index-url line as well as the three pins. Never inline versions here.
# Bumping the trio means re-running tools/BUILD_WHEELS.md first.
COPY torch/${CUDA_VARIANT}.txt /torch-trio.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /torch-trio.txt

# Freeze the torch family so no later pip install can move it
# (comfyui-wan/Dockerfile:53), then apply it as a global PIP_CONSTRAINT -
# qwen's stronger form (comfyui-qwen-image/Dockerfile:47-48). In the base this
# means every later pip install, template node layers and boot-time installs
# included, is constrained automatically, with no per-line --constraint flag
# to forget. This is what keeps the baked sage wheel's ABI safe under
# COMFYUI_VERSION=latest requirement reinstalls (plan.md 5b, torch safety).
RUN pip freeze | grep -E "^(torch|torchvision|torchaudio|torchsde)==" > /torch-constraint.txt
ENV PIP_CONSTRAINT=/torch-constraint.txt

# Core Python tooling (comfyui-wan/Dockerfile:56-57)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# Runtime libraries (comfyui-wan/Dockerfile:60-63; gdown is the contract's
# Google Drive downloader, CONTRACTS.md section 1)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pyyaml gdown triton jupyterlab jupyterlab-lsp \
        jupyter-server jupyter-server-terminals \
        ipykernel jupyterlab_code_formatter

# huggingface_hub + hf_xet for the download manager (union of
# comfyui-wan/Dockerfile:66-67 and comfyui-qwen-image/Dockerfile:59)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade huggingface_hub hf_xet

# opencv-python, hoisted out of all four template final stages
# (wan:87, minimax:123, ltx2:69, qwen:67-68). Several comfy_extras modules and
# many node packs import cv2; without it they land as IMPORT FAILED.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install opencv-python

# ----------------------------------------------------------------------------
# ComfyUI at ${COMFYUI_REF}. Shallow clone for image size, BUT with the fetch
# refspec repaired to full refs/heads/* plus tags afterwards. minimax's plain
# `git clone --depth=1 --branch ${COMFYUI_REF}` (comfyui-minimax/
# Dockerfile:104-107) leaves a single-ref refspec behind, producing a checkout
# that cannot fetch anything else - which permanently breaks ComfyUI-Manager's
# update path AND COMFYUI_VERSION=latest. Fixed here, once, for every template
# (plan.md 5b item 2).
#
# The resolved SHA goes to /comfyui-approved-ref so COMFYUI_VERSION=approved
# always has a local, network-free restore target (plan.md 5b item 1). The
# requirements install runs under PIP_CONSTRAINT set above.
# ----------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth=1 --branch ${COMFYUI_REF} \
        https://github.com/comfyanonymous/ComfyUI.git /ComfyUI \
    && git -C /ComfyUI config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*' \
    && git -C /ComfyUI config --add remote.origin.fetch '+refs/tags/*:refs/tags/*' \
    && git -C /ComfyUI rev-parse HEAD > /comfyui-approved-ref \
    && pip install -r /ComfyUI/requirements.txt

# ComfyUI-Manager, cloned as lowercase `comfyui-manager`. The lowercase name is
# kept ONLY for consistency with every existing image's custom_nodes layout.
# It does NOT control load order, despite what the template Dockerfiles claim:
# ComfyUI iterates custom_nodes with a bare os.listdir() and no sorted()
# (nodes.py:2353-2356 on v0.34.0), so load order is filesystem order,
# effectively arbitrary on overlayfs (plan.md 5c, load-order note).
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git \
        /ComfyUI/custom_nodes/comfyui-manager \
    && if [ -f /ComfyUI/custom_nodes/comfyui-manager/requirements.txt ]; then \
         pip install -r /ComfyUI/custom_nodes/comfyui-manager/requirements.txt; \
       fi

# ----------------------------------------------------------------------------
# SageAttention wheels, BOTH variants, from GitHub Release assets on
# comfyui-runtime (plan D8). Every image carries both; boot selects by
# torch.version.cuda major (plan D9). The two wheels share a filename, so they
# live in separate Release tags and separate /opt/sage/<variant>/ dirs.
#
# URLs and checksums are verbatim from the published Releases (2026-08-13,
# tools/BUILD_WHEELS.md step 12): cu128 is 35,816,068 bytes, cu130 is
# 41,362,181 bytes. These Releases are production infrastructure - never
# delete or retag them. A trio bump means re-running BUILD_WHEELS.md and
# updating both ADDs together.
# ----------------------------------------------------------------------------
ADD --checksum=sha256:487aeccc76236043c06154dfac34626692af25f520a3634c71d0bf160ab27ed6 \
    https://github.com/Hearmeman24/comfyui-runtime/releases/download/sage-d1a57a5-cu128-torch2.11.0/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl \
    /opt/sage/cu128/
ADD --checksum=sha256:77803563ccc1f3a29e52b7873c058a665aa9de4ca0827c9c8c507d07e25f64a1 \
    https://github.com/Hearmeman24/comfyui-runtime/releases/download/sage-d1a57a5-cu130-torch2.11.0/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl \
    /opt/sage/cu130/

# Variant-matched build-time sanity check (donor: comfyui-wan/Dockerfile:165-169,
# generalized to both variants): fail fast if torch's CUDA major does not match
# CUDA_VARIANT, then install the matching wheel and assert it imports. This is
# the drift gate plan.md 5c relies on: an incoherent trio/wheel pair dies in
# the base build, before any template exists that could ship it. Import-only -
# no GPU at build time; the real kernel probe runs at boot (sage_probe.py) and
# on the build pod (tools/build_sage_wheel.sh).
RUN set -e; \
    case "$CUDA_VARIANT" in \
        cu128) EXPECTED_CUDA_MAJOR=12 ;; \
        cu130) EXPECTED_CUDA_MAJOR=13 ;; \
        *) echo "CUDA_VARIANT must be cu128 or cu130, got: $CUDA_VARIANT" >&2; exit 1 ;; \
    esac; \
    EXPECTED_CUDA_MAJOR=$EXPECTED_CUDA_MAJOR python3 -c "import os, torch; \
        v = torch.version.cuda; \
        assert v.split('.')[0] == os.environ['EXPECTED_CUDA_MAJOR'], v; \
        print('torch', torch.__version__, 'cuda', v)"; \
    pip install --no-deps /opt/sage/"$CUDA_VARIANT"/sageattention-*.whl; \
    python3 -c "import sageattention; print('sageattention import OK ($CUDA_VARIANT)')"

# CivitAI downloader baked to /usr/local/bin at build time
# (comfyui-qwen-image/Dockerfile:130-133; wan/minimax/ltx2 pay this git clone
# on every boot instead, e.g. comfyui-wan/src/start.sh:144-148 - that boot
# step dies family-wide, plan.md 5c; start.sh keeps a clone only as an
# if-missing fallback for images built before this base existed).
RUN git clone --depth=1 https://github.com/Hearmeman24/CivitAI_Downloader.git /tmp/civitai-dl \
    && cp /tmp/civitai-dl/download_with_aria.py /usr/local/bin/ \
    && chmod +x /usr/local/bin/download_with_aria.py \
    && rm -rf /tmp/civitai-dl

# ----------------------------------------------------------------------------
# ORT_INDEX_ARGS, per variant. The base installs NO onnxruntime itself - node
# requirements in the template layer would shadow it anyway (last install
# wins), so the template owns the final `pip install onnxruntime-gpu
# $ORT_INDEX_ARGS` after its clone loop (plan.md 5c, onnxruntime ordering
# trap). The base owns only the index nuance as data: PyPI's onnxruntime-gpu
# links CUDA 13, so cu128 must pull from the Azure onnxruntime-cuda-12 index
# (comfyui-qwen-image/Dockerfile:122-127); cu130 uses PyPI, so the args are
# empty. ENV cannot be conditional, hence the per-variant stages below.
# ----------------------------------------------------------------------------
FROM runtime-common AS ort-cu128
ENV ORT_INDEX_ARGS="--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"

FROM runtime-common AS ort-cu130
ENV ORT_INDEX_ARGS=""

# The published image is whichever ort-* stage matches CUDA_VARIANT. No CMD
# here: each template COPYs its own src/start_script.sh and sets CMD.
FROM ort-${CUDA_VARIANT} AS final
