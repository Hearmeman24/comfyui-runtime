#!/usr/bin/env python3
"""Self-check: a provisioned pod must be internally consistent.

For every variant profile, the workflows copied to the user's ComfyUI must
declare exactly the model files the download manifest pulled, in the loader
widgets AND in each node's `properties.models`, which is what the ComfyUI
frontend's "Missing Models" dialog reads. A mismatch there tells the customer
a file is missing and offers to download it to their own PC. Subgraph
INSTANCE nodes carry their own promoted widget values separate from the
definition's inner loaders, so both levels are asserted.

Everything runs against synthetic fixtures in a tempdir: no network, no GPU,
no template repo. Modeled on comfyui-minimax/tools/test_provisioner.py.

Run: python3 tools/test_provisioner.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROV = REPO / "src" / "provisioner.py"

MIN_OK = 10 * 1024 * 1024
HF = "https://huggingface.co/Hearmeman/comfyui-template-assets/resolve/main"

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    assert cond, msg
    CHECKS += 1


# ---------------------------------------------------------------- fixtures

WALK_REGISTRY = {
    "dit_int8.safetensors": {"url": f"{HF}/dit_int8.safetensors",
                             "subdir": "diffusion_models"},
    "dit_fp8.safetensors": {"url": f"{HF}/dit_fp8.safetensors",
                            "subdir": "diffusion_models"},
    "te_int8.safetensors": {"url": f"{HF}/te_int8.safetensors",
                            "subdir": "text_encoders"},
    "vae.safetensors": {"url": f"{HF}/vae.safetensors", "subdir": "vae"},
    "sidecar.bin": {"url": f"{HF}/sidecar.bin", "subdir": "detection",
                    "auto_include_with": "vae.safetensors"},
    "baked.pth": {"url": f"{HF}/baked.pth", "subdir": "upscale_models",
                  "baked": True},
    "tiny.safetensors": {"url": f"{HF}/tiny.safetensors",
                         "subdir": "model_patches", "min_size_mb": 1},
}

WALK_PROFILES = {
    "int8": {"dit": "dit_int8.safetensors", "te": "te_int8.safetensors"},
    "fp8": {"dit": "dit_fp8.safetensors", "te": "te_int8.safetensors"},
    # alias profile: same files as fp8, kept so pods already setting it boot
    "nvfp4": {"dit": "dit_fp8.safetensors", "te": "te_int8.safetensors"},
}

WALK_TEMPLATE = {
    "provisioning_mode": "walk",
    "flags": {
        "download_test": {"folders": ["Test"],
                          "extra_models": ["tiny.safetensors"]},
    },
    "swap_groups": [{
        "env": "test_quant",
        "default": "int8",
        "flags": ["download_test"],
        "profiles": WALK_PROFILES,
    }],
    "auto_download": ["rife49.pth"],
    "image_baked": ["4xLSDIR.pth"],
}

MANAGED = {b for p in WALK_PROFILES.values() for b in p.values()}

SUBGRAPH_ID = "9f2c1a44-0000-4000-8000-abcdef012345"


def walk_workflow():
    """UI-format graph: top-level loaders, a subgraph definition with inner
    loaders, and a subgraph INSTANCE node whose promoted widgets duplicate
    the dit reference. All four surfaces must be retargeted on swap."""
    def models(*names):
        return [{"name": n, "url": WALK_REGISTRY[n]["url"],
                 "directory": WALK_REGISTRY[n]["subdir"]} for n in names]
    return {
        "nodes": [
            {"id": 1, "type": "UNETLoader",
             "widgets_values": ["dit_int8.safetensors", "default"],
             "properties": {"models": models("dit_int8.safetensors")}},
            {"id": 2, "type": "VAELoader",
             "widgets_values": ["vae.safetensors"],
             "properties": {"models": models("vae.safetensors")}},
            # subgraph INSTANCE: promoted widget values live on the instance
            {"id": 3, "type": SUBGRAPH_ID,
             "widgets_values": ["dit_int8.safetensors", 0.5],
             "properties": {"models": models("dit_int8.safetensors")}},
            {"id": 4, "type": "MarkdownNote",
             "widgets_values": ["Model links: dit_int8 / dit_fp8 variants"]},
            {"id": 5, "type": "LoraLoaderModelOnly",
             "widgets_values": ["user_lora.safetensors", 1.0]},
            {"id": 6, "type": "UpscaleLoader",
             "widgets_values": ["4xLSDIR.pth"]},
            {"id": 7, "type": "RIFE",
             "widgets_values": ["rife49.pth"]},
            {"id": 8, "type": "UpscaleLoader",
             "widgets_values": ["baked.pth"]},
        ],
        "definitions": {"subgraphs": [{
            "id": SUBGRAPH_ID,
            "nodes": [
                {"id": 10, "type": "CLIPLoader",
                 "widgets_values": ["te_int8.safetensors"],
                 "properties": {"models": models("te_int8.safetensors")}},
                {"id": 11, "type": "UNETLoader",
                 "widgets_values": ["dit_int8.safetensors"],
                 "properties": {"models": models("dit_int8.safetensors")}},
            ],
        }]},
    }


REGISTRY2 = {
    "base.safetensors": {"url": f"{HF}/base.safetensors",
                         "subdir": "checkpoints",
                         "flag": "download_base", "on_by_default": True},
    "legacy_full.safetensors": {"url": f"{HF}/legacy_full.safetensors",
                                "subdir": "diffusion_models",
                                "flag": "download_legacy", "variant": "full"},
    "legacy_fp8.safetensors": {"url": f"{HF}/legacy_fp8.safetensors",
                               "subdir": "diffusion_models",
                               "flag": "download_legacy", "variant": "fp8"},
    "gated.safetensors": {"url": f"{HF}/gated.safetensors",
                          "subdir": "text_encoders", "gated": True},
    "ic.safetensors": {"url": f"{HF}/ic.safetensors", "subdir": "loras",
                       "disable_flag": "disable_ic"},
    # flag never enabled: only its auto_include_with trigger can pull it in
    # (sidecars run after all gating, CONTRACTS.md section 4)
    "sidecar2.bin": {"url": f"{HF}/sidecar2.bin", "subdir": "detection",
                     "flag": "download_sidecar_never",
                     "auto_include_with": "base.safetensors"},
}

TEMPLATE2 = {
    "provisioning_mode": "registry",
    "flags": {
        "download_base": {"copy": ["."], "default": True},
        "download_legacy": {"copy": ["legacy"], "default": False},
    },
    "variant_env": "lightweight_fp8",
    "swap_groups": [{
        "env": "lightweight_fp8",
        "default": "false",
        "flags": ["download_legacy"],
        "profiles": {
            "false": {"dit": "legacy_full.safetensors"},
            "true": {"dit": "legacy_fp8.safetensors"},
        },
    }],
}


def registry_workflows():
    def models(*names):
        return [{"name": n, "url": REGISTRY2[n]["url"],
                 "directory": REGISTRY2[n]["subdir"]} for n in names]
    base = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple",
             "widgets_values": ["base.safetensors"],
             "properties": {"models": models("base.safetensors")}},
            # in the registry but its flag may be off: scan is informational
            {"id": 2, "type": "UNETLoader",
             "widgets_values": ["legacy_full.safetensors"],
             "properties": {"models": models("legacy_full.safetensors")}},
            {"id": 3, "type": "LoraLoaderModelOnly",
             "widgets_values": ["mystery_lora.safetensors", 1.0]},
        ],
    }
    legacy = {
        "nodes": [
            {"id": 1, "type": "UNETLoader",
             "widgets_values": ["legacy_full.safetensors"],
             "properties": {"models": models("legacy_full.safetensors")}},
            {"id": 2, "type": SUBGRAPH_ID,
             "widgets_values": ["legacy_full.safetensors"],
             "properties": {"models": models("legacy_full.safetensors")}},
        ],
        "definitions": {"subgraphs": [{
            "id": SUBGRAPH_ID,
            "nodes": [
                {"id": 10, "type": "UNETLoader",
                 "widgets_values": ["legacy_full.safetensors"],
                 "properties": {"models": models("legacy_full.safetensors")}},
            ],
        }]},
    }
    return base, legacy


# ----------------------------------------------------------------- helpers

def write_fixture(tmp: Path, template: dict, registry: dict,
                  workflows: dict) -> dict:
    """workflows: relative path -> doc. Returns the arg paths."""
    (tmp / "template.json").write_text(json.dumps(template, indent=2))
    (tmp / "registry.json").write_text(json.dumps(registry, indent=2))
    src = tmp / "workflows"
    for rel, doc in workflows.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2))
    return {"template": tmp / "template.json",
            "registry": tmp / "registry.json", "src": src}


def run_prov(paths, tmp: Path, tag: str, env: dict) -> tuple:
    """Run the provisioner as a subprocess with a clean env. Returns
    (returncode, stdout+stderr, manifest names set, manifest lines, dst)."""
    dst = tmp / f"dst-{tag}"
    manifest = tmp / f"manifest-{tag}.tsv"
    proc = subprocess.run(
        [sys.executable, str(PROV),
         "--template", str(paths["template"]),
         "--registry", str(paths["registry"]),
         "--workflows-src", str(paths["src"]),
         "--workflows-dst", str(dst),
         "--models-root", str(tmp / "models"),
         "--manifest", str(manifest)],
        env={"PATH": os.environ.get("PATH", ""), **env},
        capture_output=True, text=True,
    )
    lines = []
    if manifest.is_file():
        lines = [l for l in manifest.read_text().splitlines() if l.strip()]
    names = {l.split("\t")[1].rsplit("/", 1)[1] for l in lines}
    return proc.returncode, proc.stdout + proc.stderr, names, lines, dst


def declared(workflow_dir: Path, nameset: set, registry: dict) -> set:
    """Every managed basename the copied workflows claim to load, in widgets
    and properties.models, at top level and inside subgraph definitions.
    Ported from comfyui-minimax/tools/test_provisioner.py:46-67."""
    names = set()
    for wf in workflow_dir.rglob("*.json"):
        doc = json.loads(wf.read_text())
        groups = [doc.get("nodes", [])] + [
            sg.get("nodes", [])
            for sg in doc.get("definitions", {}).get("subgraphs", [])
        ]
        for group in groups:
            for n in group:
                for v in n.get("widgets_values") or []:
                    if isinstance(v, str) and v in nameset:
                        names.add(v)
                for m in (n.get("properties") or {}).get("models") or []:
                    if m.get("name") in nameset:
                        names.add(m["name"])
                        ok(m.get("url") == registry[m["name"]]["url"],
                           f"{wf.name}: {m['name']} declares url "
                           f"{m.get('url')}, registry says "
                           f"{registry[m['name']]['url']}")
    return names


def node_by_id(doc: dict, node_id: int, subgraph: bool = False) -> dict:
    groups = (doc.get("definitions", {}).get("subgraphs", [{}])[0].get(
        "nodes", []) if subgraph else doc.get("nodes", []))
    return next(n for n in groups if n.get("id") == node_id)


# ------------------------------------------------------------------- tests

def test_walk_profiles():
    for quant, profile in [(None, WALK_PROFILES["int8"]),
                           ("int8", WALK_PROFILES["int8"]),
                           ("fp8", WALK_PROFILES["fp8"]),
                           ("nvfp4", WALK_PROFILES["nvfp4"])]:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            paths = write_fixture(tmp, WALK_TEMPLATE, WALK_REGISTRY,
                                  {"Test/main.json": walk_workflow()})
            env = {"download_test": "true"}
            if quant is not None:
                env["test_quant"] = quant
            rc, out, names, lines, dst = run_prov(paths, tmp, "walk", env)
            ok(rc == 0, f"{quant}: rc={rc}\n{out}")
            dit = profile["dit"]
            other = ({"dit_int8.safetensors", "dit_fp8.safetensors"}
                     - {dit}).pop()
            wanted = set(profile.values()) | {
                "vae.safetensors", "sidecar.bin", "tiny.safetensors"}
            ok(names == wanted,
               f"{quant}: manifest {sorted(names)} != {sorted(wanted)}")
            ok(other not in names, f"{quant}: alternate {other} queued")
            for absent in ("baked.pth", "rife49.pth", "4xLSDIR.pth",
                           "user_lora.safetensors"):
                ok(absent not in names, f"{quant}: {absent} queued")
            # min_size_mb rides as the manifest's third field
            tiny = next(l for l in lines if "tiny.safetensors" in l)
            ok(tiny.split("\t")[2:] == ["1"], f"third field: {tiny!r}")
            # copies preserve the path relative to --workflows-src
            copy = dst / "Test" / "main.json"
            ok(copy.is_file(), f"{quant}: {copy} not copied")
            # widgets + properties.models agree with the manifest,
            # at top level AND inside the subgraph definition
            ok(declared(dst, MANAGED, WALK_REGISTRY) == set(profile.values()),
               f"{quant}: declared != profile")
            doc = json.loads(copy.read_text())
            ok(node_by_id(doc, 1)["widgets_values"][0] == dit,
               f"{quant}: top-level loader widget not {dit}")
            ok(node_by_id(doc, 3)["widgets_values"][0] == dit,
               f"{quant}: subgraph INSTANCE promoted widget not {dit}")
            ok(node_by_id(doc, 3)["properties"]["models"][0]["name"] == dit,
               f"{quant}: instance properties.models not {dit}")
            ok(node_by_id(doc, 11, subgraph=True)["widgets_values"][0] == dit,
               f"{quant}: subgraph inner loader widget not {dit}")
            ok(node_by_id(doc, 11, subgraph=True)[
                "properties"]["models"][0]["url"]
               == WALK_REGISTRY[dit]["url"],
               f"{quant}: inner properties.models url stale")
            # user-supplied ref warns, never fails
            ok("user_lora.safetensors" in out,
               f"{quant}: user-supplied warning missing\n{out}")
    print("ok: walk mode, all profiles consistent (widgets, instance "
          "promoted widgets, subgraph inner loaders, properties.models)")


def test_unknown_quant():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, WALK_TEMPLATE, WALK_REGISTRY,
                              {"Test/main.json": walk_workflow()})
        rc, out, names, _, dst = run_prov(
            paths, tmp, "bogus",
            {"download_test": "true", "test_quant": "bogus"})
        ok(rc == 0, f"unknown quant must not fail: rc={rc}\n{out}")
        ok("unknown" in out.lower(), f"no warning for bogus quant:\n{out}")
        ok("dit_int8.safetensors" in names, "bogus quant did not fall back")
        ok(declared(dst, MANAGED, WALK_REGISTRY)
           == set(WALK_PROFILES["int8"].values()), "fallback rewrite wrong")
    print("ok: unknown swap value warns and falls back to the default")


def test_flag_truthiness():
    for val, on in [("1", True), ("yes", True), ("on", True), (" TRUE ", True),
                    ("false", False), ("0", False), ("", False)]:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            paths = write_fixture(tmp, WALK_TEMPLATE, WALK_REGISTRY,
                                  {"Test/main.json": walk_workflow()})
            rc, out, names, _, _ = run_prov(paths, tmp, "truthy",
                                            {"download_test": val})
            ok(rc == 0, f"{val!r}: rc={rc}\n{out}")
            ok(bool(names) == on,
               f"opt-in flag={val!r}: expected enabled={on}, got {names}")
    print("ok: opt-in truthiness is the widened set")


def test_no_flags():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, WALK_TEMPLATE, WALK_REGISTRY,
                              {"Test/main.json": walk_workflow()})
        rc, out, names, _, dst = run_prov(paths, tmp, "none", {})
        ok(rc == 0, f"rc={rc}\n{out}")
        ok(names == set(), f"manifest not empty: {names}")
        ok((tmp / "manifest-none.tsv").read_text() == "",
           "empty manifest must be written")
        ok(not dst.exists() or not list(dst.rglob("*.json")),
           "workflows copied with no flags enabled")
        ok("no flags enabled" in out, f"one-liner missing:\n{out}")
    print("ok: no flags enabled writes an empty manifest and copies nothing")


def test_skip_on_disk():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, WALK_TEMPLATE, WALK_REGISTRY,
                              {"Test/main.json": walk_workflow()})
        env = {"download_test": "true"}
        vae = tmp / "models" / "vae" / "vae.safetensors"
        vae.parent.mkdir(parents=True)
        vae.write_bytes(b"\0" * (MIN_OK + 1))
        tiny = tmp / "models" / "model_patches" / "tiny.safetensors"
        tiny.parent.mkdir(parents=True)
        tiny.write_bytes(b"\0" * (2 * 1024 * 1024))  # above its 1 MB floor
        rc, out, names, _, _ = run_prov(paths, tmp, "disk1", env)
        ok(rc == 0, f"rc={rc}\n{out}")
        ok("vae.safetensors" not in names, "complete file re-queued")
        ok("tiny.safetensors" not in names,
           "file above its min_size_mb floor re-queued")
        vae.write_bytes(b"\0" * 10)          # truncated
        tiny.write_bytes(b"\0" * 1024)       # below its 1 MB floor
        rc, out, names, _, _ = run_prov(paths, tmp, "disk2", env)
        ok(rc == 0, f"rc={rc}\n{out}")
        ok("vae.safetensors" in names, "truncated file not re-queued")
        ok("tiny.safetensors" in names, "sub-floor small file not re-queued")
    print("ok: skip-if-on-disk honours the floor, min_size_mb included")


def test_missing_folder_warns():
    template = {"provisioning_mode": "walk",
                "flags": {"download_test": {"folders": ["Nope"]}}}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, template, WALK_REGISTRY, {})
        rc, out, names, _, _ = run_prov(paths, tmp, "nofolder",
                                        {"download_test": "true"})
        ok(rc == 0, f"missing folder must not fail: rc={rc}\n{out}")
        ok("missing" in out.lower(), f"no warning printed:\n{out}")
        ok(names == set(), f"queued from a missing folder: {names}")
    print("ok: missing workflow folder is a warning, not a failure")


def test_config_errors():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # swap profile filename absent from the registry
        bad = json.loads(json.dumps(WALK_TEMPLATE))
        bad["swap_groups"][0]["profiles"]["int8"]["dit"] = "ghost.safetensors"
        paths = write_fixture(tmp, bad, WALK_REGISTRY,
                              {"Test/main.json": walk_workflow()})
        rc, out, _, _, _ = run_prov(paths, tmp, "ghost",
                                    {"download_test": "true"})
        ok(rc == 2, f"profile file missing from registry: rc={rc}\n{out}")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, {"provisioning_mode": "bogus"},
                              WALK_REGISTRY, {})
        rc, out, _, _, _ = run_prov(paths, tmp, "mode", {})
        ok(rc == 2, f"unknown provisioning_mode: rc={rc}\n{out}")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, {}, WALK_REGISTRY, {})
        (tmp / "template.json").write_text("{not json")
        rc, out, _, _, _ = run_prov(paths, tmp, "garbage", {})
        ok(rc == 2, f"invalid template.json: rc={rc}\n{out}")
    print("ok: config errors exit 2")


def test_registry_mode():
    base, legacy = registry_workflows()
    workflows = {"base.json": base, "legacy/legacy_wf.json": legacy}

    # defaults: base on (on_by_default), legacy off, no token
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, TEMPLATE2, REGISTRY2, workflows)
        rc, out, names, _, dst = run_prov(paths, tmp, "reg-a", {})
        ok(rc == 0, f"rc={rc}\n{out}")
        ok(names == {"base.safetensors", "ic.safetensors", "sidecar2.bin"},
           f"default selection wrong: {sorted(names)}")
        ok((dst / "base.json").is_file(), "'.' copy missed base.json")
        ok(not (dst / "legacy").exists(),
           "'.' copy must be top-level only, non-recursive")
        # scan is informational in registry mode: base.json references
        # legacy_full, which is registry-known but flag-gated off
        ok("legacy_full.safetensors" not in names,
           "scan queued a flag-gated-off model in registry mode")
        ok("mystery_lora.safetensors" in out,
           f"user-supplied warning missing:\n{out}")

    # legacy on + fp8 variant + token
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, TEMPLATE2, REGISTRY2, workflows)
        env = {"download_legacy": "true", "lightweight_fp8": "true",
               "HF_TOKEN": "hf_x"}
        rc, out, names, _, dst = run_prov(paths, tmp, "reg-b", env)
        ok(rc == 0, f"rc={rc}\n{out}")
        ok(names == {"base.safetensors", "ic.safetensors", "sidecar2.bin",
                     "gated.safetensors", "legacy_fp8.safetensors"},
           f"fp8 selection wrong: {sorted(names)}")
        ok("legacy_full.safetensors" not in names, "full variant queued too")
        copy = dst / "legacy" / "legacy_wf.json"
        ok(copy.is_file(), "copy did not preserve the relative path")
        doc = json.loads(copy.read_text())
        fp8 = "legacy_fp8.safetensors"
        ok(node_by_id(doc, 1)["widgets_values"][0] == fp8,
           "top-level widget not rewritten to fp8")
        ok(node_by_id(doc, 2)["widgets_values"][0] == fp8,
           "subgraph INSTANCE promoted widget not rewritten to fp8")
        m = node_by_id(doc, 2)["properties"]["models"][0]
        ok(m["name"] == fp8 and m["url"] == REGISTRY2[fp8]["url"],
           f"instance properties.models stale: {m}")
        ok(node_by_id(doc, 10, subgraph=True)["widgets_values"][0] == fp8,
           "subgraph inner loader widget not rewritten to fp8")
        m = node_by_id(doc, 10, subgraph=True)["properties"]["models"][0]
        ok(m["name"] == fp8 and m["url"] == REGISTRY2[fp8]["url"],
           f"inner properties.models stale: {m}")
        ok(declared(dst, set(REGISTRY2), REGISTRY2) >= {fp8},
           "declared set missing the fp8 dit")

    # base explicitly off: on_by_default honours a literal "false"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, TEMPLATE2, REGISTRY2, workflows)
        rc, out, names, _, dst = run_prov(paths, tmp, "reg-c",
                                          {"download_base": "false"})
        ok(rc == 0, f"rc={rc}\n{out}")
        ok(names == {"ic.safetensors"}, f"opt-out wrong: {sorted(names)}")
        ok(not (dst / "base.json").exists(),
           "workflows copied for a disabled flag")

    # a typo must NOT strip the default set (only literal "false" does)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, TEMPLATE2, REGISTRY2, workflows)
        rc, out, names, _, _ = run_prov(paths, tmp, "reg-d",
                                        {"download_base": "no"})
        ok(rc == 0, f"rc={rc}\n{out}")
        ok("base.safetensors" in names,
           "a typo silently stripped the on_by_default set")

    # disable_flag drops its entry
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, TEMPLATE2, REGISTRY2, workflows)
        rc, out, names, _, _ = run_prov(paths, tmp, "reg-e",
                                        {"disable_ic": "true"})
        ok(rc == 0, f"rc={rc}\n{out}")
        ok("ic.safetensors" not in names, "disable_flag ignored")
    print("ok: registry mode gating, copy lists, fp8 rewrite, sidecars")


def test_variant_env_agreement():
    """The fp8/full DOWNLOAD choice and the workflow REWRITE must resolve
    the variant env var identically, whatever the customer typed. Both sides
    go through resolve_profile_key semantics: case/whitespace-insensitive
    match on the profile keys, unknown values fall back to the group default
    (full) - never to the wider opt-in TRUTHY set."""
    base, legacy = registry_workflows()
    workflows = {"base.json": base, "legacy/legacy_wf.json": legacy}
    full, fp8 = "legacy_full.safetensors", "legacy_fp8.safetensors"
    for val, want in [("true", fp8), ("TRUE", fp8), ("True", fp8),
                      (" true ", fp8), ("yes", full), ("false", full),
                      (None, full)]:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            paths = write_fixture(tmp, TEMPLATE2, REGISTRY2, workflows)
            env = {"download_legacy": "true"}
            if val is not None:
                env["lightweight_fp8"] = val
            rc, out, names, _, dst = run_prov(paths, tmp, "variant", env)
            ok(rc == 0, f"{val!r}: rc={rc}\n{out}")
            queued = names & {full, fp8}
            ok(queued == {want},
               f"{val!r}: queued {sorted(queued)}, want {want}")
            doc = json.loads((dst / "legacy" / "legacy_wf.json").read_text())
            widget = node_by_id(doc, 1)["widgets_values"][0]
            ok(widget == want,
               f"{val!r}: workflows load {widget} but manifest queued "
               f"{sorted(queued)}")
    print("ok: variant download and workflow rewrite resolve the env "
          "identically (case/whitespace tolerant, typo falls back to full)")


def test_deprecated_flag():
    """A retired flag must stay ACCEPTED: warn, ship nothing, never break a boot.

    CLAUDE.md section 3: "Never remove a registry entry that an env value maps
    to; keep the value accepted." A customer with download_ltx2_19b=true saved
    in their RunPod template must get a clear line and a working pod, not a
    dead one.
    """
    registry = {
        "keep.safetensors": {"url": "https://h/k", "subdir": "vae",
                             "flag": "download_new"},
        "retired.safetensors": {"url": "https://h/r", "subdir": "vae",
                                "flag": "download_old"},
    }
    template = {
        "provisioning_mode": "registry",
        "flags": {"download_new": {"copy": ["new.json"]}},
        "deprecated_flags": {
            "download_old": "LTX-2 19B is retired; use download_ltx25 instead. "
                            "Models already on your volume are untouched."
        },
    }
    wf = {"new.json": {"nodes": [{"id": 1, "type": "L",
                                  "widgets_values": ["keep.safetensors"]}]}}

    # Set the retired flag: warns, ships nothing from it, still exits 0.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, template, registry, wf)
        rc, out, names, _, dst = run_prov(
            paths, tmp, "dep-on",
            {"download_new": "true", "download_old": "true"})
        ok(rc == 0, f"a retired flag must never fail the boot: rc={rc}\n{out}")
        ok("retired" in out.lower() or "deprecat" in out.lower(),
           f"the deprecation must be announced: {out}")
        ok("download_ltx25" in out,
           f"the message must name the replacement: {out}")
        ok("retired.safetensors" not in names,
           f"a retired flag must queue nothing: {sorted(names)}")
        ok("keep.safetensors" in names,
           f"other flags must be unaffected: {sorted(names)}")

    # Not set: no noise at all.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        paths = write_fixture(tmp, template, registry, wf)
        rc, out, names, _, _ = run_prov(paths, tmp, "dep-off",
                                        {"download_new": "true"})
        ok(rc == 0, f"rc={rc}\n{out}")
        ok("download_ltx25" not in out,
           f"an unset retired flag must print nothing: {out}")
        ok("retired.safetensors" not in names, "still never queued")
    print("ok: a deprecated flag is accepted, announced, and ships nothing")


def test_selftest():
    proc = subprocess.run([sys.executable, str(PROV), "--selftest"],
                          capture_output=True, text=True,
                          env={"PATH": os.environ.get("PATH", "")})
    ok(proc.returncode == 0,
       f"--selftest rc={proc.returncode}\n{proc.stdout}{proc.stderr}")
    ok("ok" in proc.stdout, f"--selftest output: {proc.stdout!r}")
    print("ok: provisioner --selftest passes")


def main() -> int:
    test_walk_profiles()
    test_unknown_quant()
    test_flag_truthiness()
    test_no_flags()
    test_skip_on_disk()
    test_missing_folder_warns()
    test_config_errors()
    test_registry_mode()
    test_variant_env_agreement()
    test_deprecated_flag()
    test_selftest()
    print(f"all provisioner checks passed ({CHECKS} assertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
