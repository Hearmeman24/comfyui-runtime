#!/usr/bin/env python3
"""Single source of truth for the base image: versions.json.

Every other place that names the ComfyUI pin is derived from, or checked
against, versions.json:

  versions.py tag <variant>        image tag, e.g. cu130-comfy0.42.1-63be2f9b-torch2.11.0
  versions.py image <variant>      hearmeman/comfyui-base:<tag>
  versions.py build-args <variant> --build-arg flags for base/Dockerfile (one per line)
  versions.py check                fail if base/Dockerfile ARG defaults or
                                   torch/<variant>.txt disagree with versions.json
  versions.py sync                 rewrite the base/Dockerfile ARG defaults from
                                   versions.json (used by the release watcher)
  versions.py set-release <tag> <sha>
                                   update versions.json's ComfyUI pin, then sync

The label is derived (<release without v>-<sha[:8]>), never stored, so it
cannot drift from the sha. Stdlib only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "versions.json"
DOCKERFILE = ROOT / "base" / "Dockerfile"
IMAGE_REPO = "hearmeman/comfyui-base"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RELEASE_RE = re.compile(r"^v\d+\.\d+\.\d+$")


class VersionsError(Exception):
    pass


def load(path: Path = VERSIONS) -> dict:
    data = json.loads(path.read_text())
    comfy = data.get("comfyui", {})
    if not RELEASE_RE.match(comfy.get("release", "")):
        raise VersionsError(f"comfyui.release must look like v0.42.1, got {comfy.get('release')!r}")
    if not SHA_RE.match(comfy.get("sha", "")):
        raise VersionsError("comfyui.sha must be a full 40-character commit sha")
    if not data.get("torch"):
        raise VersionsError("torch is missing")
    variants = data.get("cuda_variants") or {}
    if data.get("template_variant") not in variants:
        raise VersionsError("template_variant must be one of cuda_variants")
    return data


def label(data: dict) -> str:
    return f"{data['comfyui']['release'].lstrip('v')}-{data['comfyui']['sha'][:8]}"


def tag(data: dict, variant: str) -> str:
    if variant not in data["cuda_variants"]:
        raise VersionsError(f"unknown variant {variant!r}")
    return f"{variant}-comfy{label(data)}-torch{data['torch']}"


def image(data: dict, variant: str) -> str:
    return f"{IMAGE_REPO}:{tag(data, variant)}"


def build_args(data: dict, variant: str) -> list[str]:
    tag(data, variant)  # validates the variant
    return [
        f"--build-arg=CUDA_VARIANT={variant}",
        f"--build-arg=CUDA_BASE_IMAGE={data['cuda_variants'][variant]}",
        f"--build-arg=COMFYUI_REF={data['comfyui']['sha']}",
        f"--build-arg=COMFYUI_IMAGE_REF={label(data)}",
    ]


def _dockerfile_arg(text: str, name: str) -> str | None:
    m = re.search(rf"(?m)^ARG {name}=(\S+)$", text)
    return m.group(1) if m else None


def check(data: dict, root: Path = ROOT) -> list[str]:
    problems = []
    text = (root / "base" / "Dockerfile").read_text()
    for name, want in (("COMFYUI_REF", data["comfyui"]["sha"]), ("COMFYUI_IMAGE_REF", label(data))):
        got = _dockerfile_arg(text, name)
        if got != want:
            problems.append(f"base/Dockerfile ARG {name}={got} but versions.json says {want} (run tools/versions.py sync)")
    for variant in data["cuda_variants"]:
        trio = root / "torch" / f"{variant}.txt"
        if not trio.exists():
            problems.append(f"torch/{variant}.txt missing for cuda variant {variant}")
            continue
        m = re.search(r"(?m)^torch==(\S+)$", trio.read_text())
        if not m or m.group(1) != data["torch"]:
            problems.append(f"torch/{variant}.txt pins torch=={m.group(1) if m else '?'} but versions.json says {data['torch']}")
    return problems


def sync(data: dict, root: Path = ROOT) -> bool:
    path = root / "base" / "Dockerfile"
    text = path.read_text()
    new = re.sub(r"(?m)^ARG COMFYUI_REF=\S+$", f"ARG COMFYUI_REF={data['comfyui']['sha']}", text)
    new = re.sub(r"(?m)^ARG COMFYUI_IMAGE_REF=\S+$", f"ARG COMFYUI_IMAGE_REF={label(data)}", new)
    new = re.sub(r"(?m)^# Latest stable release v\S+ \([^)]*\), resolved to an immutable SHA\.$",
                 f"# Latest stable release {data['comfyui']['release']}, resolved to an immutable SHA (versions.json).", new)
    if new != text:
        path.write_text(new)
    return new != text


def set_release(release: str, sha: str, path: Path = VERSIONS) -> dict:
    raw = json.loads(path.read_text())
    raw["comfyui"] = {"release": release, "sha": sha}
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return load(path)


def main(argv: list[str]) -> int:
    try:
        if not argv:
            raise VersionsError(__doc__)
        cmd, rest = argv[0], argv[1:]
        if cmd == "set-release" and len(rest) == 2:
            data = set_release(*rest)
            sync(data)
            print(label(data))
            return 0
        data = load()
        if cmd == "tag" and len(rest) == 1:
            print(tag(data, rest[0]))
        elif cmd == "image" and len(rest) == 1:
            print(image(data, rest[0]))
        elif cmd == "build-args" and len(rest) == 1:
            print("\n".join(build_args(data, rest[0])))
        elif cmd == "check" and not rest:
            problems = check(data)
            for p in problems:
                print(f"FAIL: {p}", file=sys.stderr)
            if problems:
                return 1
            print(f"versions.json coherent: {label(data)}, torch {data['torch']}, variants {', '.join(data['cuda_variants'])}")
        elif cmd == "sync" and not rest:
            print("synced" if sync(data) else "already in sync")
        else:
            raise VersionsError(f"bad command: {' '.join(argv)}\n{__doc__}")
    except (VersionsError, KeyError, ValueError) as exc:
        print(f"versions.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
