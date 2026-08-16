#!/usr/bin/env python3
"""Offline self-test for tools/validate_models.py.

Run: python3 tools/test_validate_models.py

No network is ever touched: the module's single seam (http_request) is
replaced with a fake that serves canned HF tree listings and ranged-GET
responses, and records every call so the tests can assert what was (and
was not) requested. Stdlib only, no pytest.
"""
from __future__ import annotations  # PEP 604 syntax under python < 3.10
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_models as vm  # noqa: E402

# Every other test swaps vm.http_request out for a FakeNet, so keep the real
# one from before any of that happens: the bearer-header tests drive it.
REAL_HTTP_REQUEST = vm.http_request

CHECKS: list[tuple[bool, str]] = []


def check(cond, label):
    CHECKS.append((bool(cond), label))
    print(("ok   " if cond else "FAIL ") + label)


class FakeNet:
    """Drop-in replacement for vm.http_request."""

    def __init__(self):
        self.calls = []  # (url, range_first_byte)
        self.routes = {}  # url -> (status, headers, body, final_url)

    def add(self, url, status=200, headers=None, body=b"", final_url=None):
        self.routes[url] = (status, dict(headers or {}), body, final_url or url)

    def add_tree(self, repo, rev, dirpath, entries, landed=None, url_suffix="",
                 next_link=None):
        """Serve a HF tree API listing: entries = [(path_in_repo, size), ...]."""
        url = f"https://huggingface.co/api/models/{repo}/tree/{rev}"
        if dirpath:
            url += f"/{dirpath}"
        url += url_suffix
        body = json.dumps([{"type": "file", "path": p, "size": s}
                           for p, s in entries]).encode()
        final = url if landed is None else url.replace(
            f"/api/models/{repo}/", f"/api/models/{landed}/")
        headers = {"Link": f'<{next_link}>; rel="next"'} if next_link else {}
        self.add(url, 200, headers, body, final)

    def __call__(self, url, range_first_byte=False):
        self.calls.append((url, range_first_byte))
        if url not in self.routes:
            raise AssertionError(f"unexpected network call: {url}")
        return self.routes[url]


def wf(top=None, sub=None):
    """A minimal workflow doc with loader widgets at the top level and/or
    inside a subgraph definition."""
    doc = {"nodes": [{"type": "Loader", "widgets_values": [v]} for v in (top or [])]}
    if sub is not None:
        doc["definitions"] = {"subgraphs": [
            {"nodes": [{"type": "Loader", "widgets_values": [v]} for v in sub]}]}
    return doc


def make_workflows(root, files):
    wdir = Path(root) / "workflows"
    for rel, doc in files.items():
        p = wdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return wdir


def hf_entry(name, subdir, repo="org/repo", rev="main", rdir="", **extra):
    path = f"{rdir}/{name}" if rdir else name
    return {"url": f"https://huggingface.co/{repo}/resolve/{rev}/{path}",
            "subdir": subdir, **extra}


# --- gap 1: subgraph definitions are walked -------------------------------

def test_subgraph_walk():
    reg = {"sub_vae.safetensors": hf_entry("sub_vae.safetensors", "vae", rdir="vae")}
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"OnlySub.json": wf(
            top=[], sub=["vae/sub_vae.safetensors", "sub_mystery.safetensors"])})
        errors, warnings = vm.check_coverage(reg, wdir)
    check(any("wrong folder prefix" in e and "sub_vae" in e for e in errors),
          "gap 1: prefix error is found inside a subgraph-only workflow")
    check(any("sub_mystery.safetensors" in e for e in errors),
          "gap 1: the unshipped-model ERROR is found inside a subgraph-only workflow")
    check(not any("sub_mystery" in w for w in warnings),
          "severity: a loader naming an unshipped model is an ERROR, not a warning (E13)")


# --- E13: the gate that stops a personal LoRA leaking into a workflow ------

def test_unshipped_loader_ref_is_an_error():
    """The whole point of E13. A loader pointing at something the template
    does not ship promises a model it never delivers, and that is how 28 dead
    references accumulated in wan before the 2026-08-14 cull."""
    reg = {"shipped.safetensors": hf_entry("shipped.safetensors", "loras")}
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"W.json": wf(
            top=["shipped.safetensors", "deepthroat_epoch_80.safetensors"])})
        errors, warnings = vm.check_coverage(reg, wdir)
    check(any("deepthroat_epoch_80.safetensors" in e and "does not ship" in e
              for e in errors),
          "E13: a leaked personal LoRA in a loader is a hard error")
    check(any("placeholder" in e for e in errors),
          "E13: the error tells you how to fix it")
    check(not any("shipped.safetensors" in e for e in errors),
          "E13: a registered model is untouched")


def test_placeholders_are_allowed():
    """Every shipped workflow has empty LoRA slots on purpose. If those
    errored, the gate could not be switched on at all."""
    reg = {"shipped.safetensors": hf_entry("shipped.safetensors", "loras")}
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"W.json": wf(top=sorted(vm.PLACEHOLDERS))})
        errors, warnings = vm.check_coverage(reg, wdir)
    check(not errors, f"E13: placeholders never error (got {errors})")
    check(not warnings, f"E13: placeholders are not warned about either (got {warnings})")


def test_typo_placeholder_still_errors():
    """An explicit set, not a Your_* prefix match: a near-miss must fail loudly
    rather than be waved through as 'close enough to a placeholder'."""
    reg = {}
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"W.json": wf(
            top=["Your_Charcter_LoRA_Here.safetensors"])})
        errors, _ = vm.check_coverage(reg, wdir)
    check(any("Your_Charcter" in e for e in errors),
          "E13: a misspelt placeholder is not silently accepted")


# --- extra.prompt: an inert API snapshot must not be scanned ---------------

def test_extra_prompt_is_not_scanned():
    """Comfy-Org ships its LTX-2.5 templates with a stale extra.prompt naming
    models the live graph does not use. ComfyUI executes doc["nodes"]; this
    snapshot is never read, so reporting it trains people to ignore warnings."""
    reg = {"real.safetensors": hf_entry("real.safetensors", "vae")}
    doc = wf(top=["real.safetensors"])
    doc["extra"] = {"ds": {"scale": 1},
                    "prompt": {"1": {"class_type": "CheckpointLoaderSimple",
                                     "inputs": {"ckpt_name": "ghost_from_an_old_graph.safetensors"}}}}
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"W.json": doc})
        errors, warnings = vm.check_coverage(reg, wdir)
    check(not any("ghost_from_an_old_graph" in m for m in errors + warnings),
          "extra.prompt is excluded from the raw-text scan")
    check(not errors, f"a workflow that is fine stays clean (got {errors})")


def test_raw_text_scan_still_runs_outside_extra_prompt():
    """Pruning extra.prompt must not disable the scan. properties.models is
    exactly where the ltx2 Face-ID metadata bug hid, and it stays warned."""
    reg = {"real.safetensors": hf_entry("real.safetensors", "vae")}
    doc = wf(top=["real.safetensors"])
    doc["nodes"][0]["properties"] = {
        "models": [{"name": "stale_hint.safetensors", "url": "https://x/y", "directory": "vae"}]}
    doc["extra"] = {"prompt": {"1": {"inputs": {"ckpt_name": "ghost.safetensors"}}}}
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"W.json": doc})
        errors, warnings = vm.check_coverage(reg, wdir)
    check(any("stale_hint.safetensors" in w for w in warnings),
          "properties.models is still scanned, at warning level")
    check(not any("stale_hint" in e for e in errors),
          "properties.models stays a warning: it is a hint, not what the loader loads")
    check(not any("ghost" in m for m in errors + warnings),
          "and extra.prompt is still excluded in the same file")


# --- gap 2: the full widget value is compared, prefix included ------------

def test_widget_prefix():
    reg = {
        "foo_vae.safetensors": hf_entry("foo_vae.safetensors", "vae"),
        "bar_lora.safetensors": hf_entry("bar_lora.safetensors", "loras/ltx2"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {
            "Stray.json": wf(top=["vae/foo_vae.safetensors"]),
            "Good.json": wf(top=["ltx2/bar_lora.safetensors"]),
            "Bare.json": wf(top=["bar_lora.safetensors"]),
        })
        errors, warnings = vm.check_coverage(reg, wdir)
    check(any("Stray.json" in e and "vae/foo_vae.safetensors" in e
              and "'foo_vae.safetensors'" in e for e in errors),
          "gap 2: stray 'vae/' prefix on a category-root model is an error")
    check(not any("Good.json" in e for e in errors),
          "gap 2: correct nested prefix (loras/ltx2 -> 'ltx2/') passes")
    check(any("Bare.json" in e and "'ltx2/bar_lora.safetensors'" in e for e in errors),
          "gap 2: missing required prefix on a nested-subdir model is an error")
    check(not warnings, "gap 2: no spurious user-supplied warnings")


# --- gap 3: gated entries use the tree listing, never a bare HEAD ---------

def test_gated_tree_api():
    fake = FakeNet()
    vm.http_request = fake
    reg = {
        "gated_ok.safetensors": hf_entry("gated_ok.safetensors", "text_encoders",
                                         repo="org/gated", rdir="tenc", gated=True),
        "gated_gone.safetensors": hf_entry("gated_gone.safetensors", "text_encoders",
                                           repo="org/gated", rdir="tenc", gated=True),
    }
    fake.add_tree("org/gated", "main", "tenc",
                  [("tenc/gated_ok.safetensors", 5_000_000_000)])
    errors, _ = vm.check_urls(reg)
    check(not any("gated_ok" in e for e in errors),
          "gap 3: present file in a gated repo passes via the tree listing")
    check(any("gated_gone" in e and "removed or renamed" in e for e in errors),
          "gap 3: deleted file in a gated repo is a hard error")
    check(all("/api/models/" in u and not rng for u, rng in fake.calls),
          "gap 3: only the model API was queried; no request on the resolve URL")


def test_renamed_repo_named_in_error():
    fake = FakeNet()
    vm.http_request = fake
    reg = {"gone.safetensors": hf_entry("gone.safetensors", "vae",
                                        repo="org/old", rdir="vae")}
    fake.add_tree("org/old", "main", "vae",
                  [("vae/other.safetensors", 900_000_000)], landed="org/newname")
    errors, _ = vm.check_urls(reg)
    check(any("renamed to org/newname" in e for e in errors),
          "gap 3: a repo rename is followed and named in the error")


def test_tree_pagination():
    fake = FakeNet()
    vm.http_request = fake
    page2 = "https://huggingface.co/api/models/org/big/tree/main/vae?cursor=abc"
    fake.add_tree("org/big", "main", "vae",
                  [("vae/first.safetensors", 900_000_000)], next_link=page2)
    fake.add(page2, 200, {}, json.dumps(
        [{"type": "file", "path": "vae/second.safetensors", "size": 900_000_000}]).encode())
    reg = {"second.safetensors": hf_entry("second.safetensors", "vae",
                                          repo="org/big", rdir="vae")}
    errors, _ = vm.check_urls(reg)
    check(not errors, "gap 3: tree listing pagination is followed")


# --- gap 4: non-HF URLs get a ranged GET, never a HEAD --------------------

def test_ranged_get():
    url = "https://r2.example.com/private/client_lora.safetensors?X-Amz-Signature=abc"
    reg = {"client_lora.safetensors": {"url": url, "subdir": "loras"}}

    fake = FakeNet()
    vm.http_request = fake
    fake.add(url, 206, {"Content-Range": "bytes 0-0/123456789"}, b"\x00")
    errors, warnings = vm.check_urls(reg)
    check(not errors and not warnings,
          "gap 4: presigned URL passes on 206 with Content-Range")
    check(fake.calls == [(url, True)],
          "gap 4: exactly one ranged GET (Range: bytes=0-0) was issued")

    fake = FakeNet()
    vm.http_request = fake
    fake.add(url, 403, {}, b"")
    errors, _ = vm.check_urls(reg)
    check(any("HTTP 403 on ranged GET" in e for e in errors),
          "gap 4: a failing ranged GET is a hard error (exit non-zero path)")


# --- size sanity -----------------------------------------------------------

def test_size_sanity():
    onnx = hf_entry("graph.onnx", "detection", rdir="det")

    fake = FakeNet()
    vm.http_request = fake
    fake.add_tree("org/repo", "main", "det", [("det/graph.onnx", 430_080)])
    errors, warnings = vm.check_urls({"graph.onnx": onnx})
    check(not errors and any("implausibly small" in w and "auto_include_with" in w
                             for w in warnings),
          "size: a 0.41 MB entry with no declared sidecar warns")

    reg2 = {
        "graph.onnx": onnx,
        "graph_data.bin": hf_entry("graph_data.bin", "detection", rdir="det",
                                   auto_include_with="graph.onnx",
                                   min_size_mb=2000),
    }
    fake = FakeNet()
    vm.http_request = fake
    fake.add_tree("org/repo", "main", "det",
                  [("det/graph.onnx", 430_080), ("det/graph_data.bin", 2_500_000_000)])
    errors, warnings = vm.check_urls(reg2)
    check(not errors and not any("implausibly small" in w for w in warnings),
          "size: an auto_include_with sidecar suppresses the warning")

    fake = FakeNet()
    vm.http_request = fake
    fake.add_tree("org/repo", "main", "det", [("det/graph.onnx", 430_080)])
    errors, warnings = vm.check_urls({"graph.onnx": {**onnx, "min_size_mb": 0.3}})
    check(not errors and not warnings,
          "size: an entry whose min_size_mb vouches for the size is clean")

    fake = FakeNet()
    vm.http_request = fake
    fake.add_tree("org/repo", "main", "det", [("det/graph.onnx", 430_080)])
    errors, _ = vm.check_urls({"graph.onnx": {**onnx, "min_size_mb": 500}})
    check(any("below its own min_size_mb" in e for e in errors),
          "size: remote size below the declared min_size_mb floor is a hard error")


# --- schema: dest_subdir is rejected ---------------------------------------

def test_dest_subdir_rejected():
    errors = vm.check_registry_schema(
        {"x.safetensors": {"url": "https://example.com/x", "dest_subdir": "models/vae"}})
    check(any("dest_subdir" in e for e in errors),
          "schema: dest_subdir is rejected (CONTRACTS.md section 4)")
    check(any("missing required field 'subdir'" in e for e in errors),
          "schema: url and subdir are required")


# --- baked entries skip the network ----------------------------------------

def test_baked_skips_network():
    fake = FakeNet()  # no routes: any call raises
    vm.http_request = fake
    reg = {"4xLSDIR.pth": {"url": "https://example.com/4xLSDIR.pth",
                           "subdir": "upscale_models", "baked": True}}
    errors, warnings = vm.check_urls(reg)
    check(not errors and not warnings and not fake.calls,
          "baked: image-baked entries are never checked upstream")


# --- template.json cross-checks ---------------------------------------------

def test_template_checks():
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {
            "MiniMax H3/A.json": wf(top=[]),
            "LTX2.5/B.json": wf(top=[]),
        })
        template = {
            "flags": {
                "download_a": {"workflows": ["Missing.json"]},
                "download_b": {"folders": ["Nope", "MiniMax H3"]},
                "download_c": {"copy": [".", "LTX2.5", "gone_dir"]},
            },
            "swap_groups": [{"env": "minimax_quant", "profiles": {
                "int8": {"dit": "not_in_registry.safetensors"}}}],
        }
        errors, warnings = vm.check_template(template, {}, wdir)
    check(any("Missing.json" in e for e in errors),
          "template: a flag naming a missing workflow file is an error")
    check(any("'Nope'" in w for w in warnings) and
          not any("MiniMax H3" in w for w in warnings),
          "template: a missing flag folder is only a warning")
    check(any("'gone_dir'" in e for e in errors) and
          not any("'LTX2.5'" in e or "'.'" in e for e in errors),
          "template: a missing copy path is an error; '.' and existing dirs pass")
    check(any("not_in_registry.safetensors" in e for e in errors),
          "template: a swap profile filename absent from the registry is an error")


# --- E16: the template.json key allowlist -----------------------------------

def valid_template(mode="walk"):
    """A template of each real shape, exercising every allowed key."""
    if mode == "walk":
        return {
            "template_repo": "https://github.com/Hearmeman24/comfyui-wan.git",
            "branch": "master",
            "provisioning_mode": "walk",
            "flags": {"download_wan21": {"folders": ["Wan 2.1"],
                                         "extra_models": ["rife426.pth"],
                                         "default": True},
                      "DOWNLOAD_QWEN_IMAGE": {"workflows": ["Q.json"]}},
            "swap_groups": [{"env": "minimax_quant", "default": "int8",
                             "flags": ["download_wan21"],
                             "profiles": {"int8": {"dit": "a.safetensors"},
                                          "fp8": {"dit": "b.safetensors"}}}],
            "variant_env": "lightweight_fp8",
            "deprecated_flags": {"download_old": "use download_wan21 instead."},
            "auto_download": ["rife49.pth"],
            "image_baked": ["4xLSDIR.pth"],
            "extra_model_paths": ["vae_approx"],
            "models_symlink": False,
            "custom_nodes": {"target": "image", "repos": ["https://x/y.git|abc"]},
            "sage": True,
        }
    return {
        "template_repo": "https://github.com/Hearmeman24/comfyui-ltx2.git",
        "branch": "main",
        "provisioning_mode": "registry",
        "flags": {"download_ltx23": {"copy": ["."], "default": True}},
        "custom_nodes": {"target": "volume", "repos": []},
        "sage": True,
    }


def test_template_schema_accepts_both_real_shapes():
    check(vm.check_template_schema(valid_template("walk")) == [],
          "E16: a valid walk-mode template with every allowed key passes")
    check(vm.check_template_schema(valid_template("registry")) == [],
          "E16: a valid registry-mode template passes")


def test_template_schema_accepts_entrypoint_keys():
    """template_repo and branch are read by the template's baked
    start_script.sh, not by the runtime, so deriving the allowlist from a scan
    of runtime code alone would turn all four templates red."""
    t = {"provisioning_mode": "walk",
         "template_repo": "https://github.com/Hearmeman24/comfyui-wan.git",
         "branch": "master"}
    check(vm.check_template_schema(t) == [],
          "E16: template_repo and branch are accepted, not flagged as unknown")


def test_template_schema_accepts_comfy_extra_args():
    """minimax carries --disable-dynamic-vram here to dodge an open upstream
    bug (Comfy-Org/ComfyUI#15271). If this key ever falls out of the allowlist
    the template goes red in CI, and the crash it prevents comes straight back."""
    t = {"provisioning_mode": "walk",
         "comfy_extra_args": "--disable-dynamic-vram"}
    check(vm.check_template_schema(t) == [],
          "E16: comfy_extra_args is an allowed top-level key")
    check("comfy_extra_args" in vm.TEMPLATE_KEYS,
          "E16: and it is in TEMPLATE_KEYS, not accepted by accident")


def test_template_schema_rejects_unknown_top_level_key():
    """Measured against wan: 'flags' -> 'flag' disables the ENTIRE template and
    still exits 0 everywhere (EXECUTION.md E16)."""
    t = valid_template("walk")
    t["flag"] = t.pop("flags")
    errors = vm.check_template_schema(t)
    check(any("unknown key 'flag'" in e and "top level" in e for e in errors),
          "E16: an unknown top-level key errors and is named")
    t2 = dict(valid_template("walk"), auto_downloads=["x.pth"])
    check(any("unknown key 'auto_downloads'" in e
              for e in vm.check_template_schema(t2)),
          "E16: 'auto_download' -> 'auto_downloads' is caught by name, not by "
          "8 errors that blame the workflows")


def test_template_schema_rejects_unknown_flag_key():
    """'folders' -> 'folder' copies zero workflows: the customer enables a flag,
    pays for a GPU and gets an empty ComfyUI."""
    t = valid_template("walk")
    t["flags"]["download_wan21"]["folder"] = t["flags"]["download_wan21"].pop("folders")
    errors = vm.check_template_schema(t)
    check(any("unknown key 'folder'" in e and "flag 'download_wan21'" in e
              for e in errors),
          "E16: an unknown per-flag key errors and names the flag")
    t2 = valid_template("walk")
    t2["flags"]["download_wan21"]["extra_model"] = \
        t2["flags"]["download_wan21"].pop("extra_models")
    check(any("unknown key 'extra_model'" in e for e in vm.check_template_schema(t2)),
          "E16: 'extra_models' -> 'extra_model' no longer drops a model silently")


def test_template_schema_rejects_unknown_swap_group_and_profile_shape():
    t = valid_template("walk")
    t["swap_groups"][0]["profile"] = t["swap_groups"][0].pop("profiles")
    errors = vm.check_template_schema(t)
    check(any("unknown key 'profile'" in e and "swap group 'minimax_quant'" in e
              for e in errors),
          "E16: an unknown swap-group key errors and names the group")
    t2 = valid_template("walk")
    t2["swap_groups"][0]["profiles"]["int8"]["dit"] = ["a.safetensors"]
    check(any("role 'dit' must be a filename string" in e
              for e in vm.check_template_schema(t2)),
          "E16: a profile role must map to a filename string, not a container")
    t3 = valid_template("walk")
    t3["custom_nodes"]["repo"] = t3["custom_nodes"].pop("repos")
    check(any("unknown key 'repo'" in e and "custom_nodes" in e
              for e in vm.check_template_schema(t3)),
          "E16: an unknown key inside custom_nodes errors")


def test_template_schema_runs_in_the_gate():
    """The allowlist must fire through run(), not just as a library call:
    that is the only path CI takes."""
    def boom(url, range_first_byte=False):
        raise AssertionError(f"network call in offline mode: {url}")
    vm.http_request = boom
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "models_registry.json"
        reg_path.write_text(json.dumps({}))
        wdir = make_workflows(tmp, {"Good.json": wf(top=[])})
        tpl = Path(tmp) / "template.json"
        tpl.write_text(json.dumps({"provisioning_mode": "walk", "flags": {},
                                   "template_repo": "x", "branch": "master"}))
        rc = vm.run(reg_path, wdir, template_path=tpl, offline=True)
        check(rc == 0, "E16: a schema-clean template still exits 0 through run()")
        tpl.write_text(json.dumps({"provisioning_mode": "walk", "flag": {},
                                   "template_repo": "x", "branch": "master"}))
        rc = vm.run(reg_path, wdir, template_path=tpl, offline=True)
        check(rc == 1, "E16: a typo'd template.json key exits 1 through run()")


# --- credential redaction in error strings ----------------------------------

def test_presigned_url_is_redacted_in_errors():
    """A presigned R2/S3 URL carries a live signature in its query string, and
    these messages go straight into CircleCI job output."""
    sig = "X-Amz-Signature=deadbeefliveCREDENTIAL"
    url = f"https://r2.example.com/private/client_lora.safetensors?{sig}"
    reg = {"client_lora.safetensors": {"url": url, "subdir": "loras"}}

    fake = FakeNet()
    vm.http_request = fake
    fake.add(url, 403, {}, b"")
    errors, _ = vm.check_urls(reg)
    check(errors and not any(sig in e or "X-Amz" in e for e in errors),
          "redaction: a failed ranged GET never prints the signature")
    check(any("https://r2.example.com/private/client_lora.safetensors" in e
              for e in errors),
          "redaction: host and path survive, which is what makes the error useful")

    def raiser(u, range_first_byte=False):
        raise RuntimeError(f"connection reset while fetching {u}")
    vm.http_request = raiser
    errors, _ = vm.check_urls(reg)
    check(errors and not any(sig in e for e in errors),
          "redaction: a transport error that quotes the URL back is redacted too")
    check(vm.redact("https://x/y.safetensors") == "https://x/y.safetensors",
          "redaction: a URL with no query string is unchanged")


def test_allowlists_suppress_warnings():
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {"W.json": wf(top=["rife49.pth", "4xLSDIR.pth"])})
        _, warnings = vm.check_coverage({}, wdir,
                                        allow=frozenset({"rife49.pth", "4xLSDIR.pth"}))
    check(not warnings,
          "template: auto_download / image_baked names do not warn")


# --- end to end: offline run, exit codes ------------------------------------

def test_offline_and_exit_codes():
    def boom(url, range_first_byte=False):
        raise AssertionError(f"network call in offline mode: {url}")
    vm.http_request = boom
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "models_registry.json"
        reg_path.write_text(json.dumps(
            {"foo_vae.safetensors": hf_entry("foo_vae.safetensors", "vae")}))
        wdir = make_workflows(tmp, {"Good.json": wf(top=["foo_vae.safetensors"])})
        rc = vm.run(reg_path, wdir, offline=True)
        check(rc == 0, "e2e: clean fixture exits 0 offline with zero network calls")
        (wdir / "Bad.json").write_text("{not json")
        rc = vm.run(reg_path, wdir, offline=True)
        check(rc == 1, "e2e: an unparseable workflow exits 1")
        rc = vm.main(["--registry", str(reg_path), "--workflows", str(wdir / "nope"),
                      "--offline"])
        check(rc == 1, "e2e: a missing workflows dir exits 1 through main()")


def test_prose_is_not_a_filename():
    """A note whose text merely ENDS in a model URL is not a model reference.

    A real wan workflow ships a note closing with a download link, so the whole
    multi-line blob ended in ".safetensors", the coverage check derived a folder
    prefix from a paragraph, and it errored, blocking the build. Filenames
    containing SPACES must still be checked: skipping those would swap a loud
    false positive for a silent false negative.
    """
    prose = ("Download models:\n\nvae:\n"
             "https://huggingface.co/org/repo/resolve/main/foo_vae.safetensors")
    bare_url = "https://huggingface.co/org/repo/resolve/main/foo_vae.safetensors"
    reg = {
        "foo_vae.safetensors": hf_entry("foo_vae.safetensors", "vae"),
        "my favourite extremely long descriptive lora name rank 105 bf16.safetensors": hf_entry("my favourite extremely long descriptive lora name rank 105 bf16.safetensors", "loras"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {
            "Note.json": wf(top=[prose]),
            "Url.json": wf(top=[bare_url]),
            "Spaced.json": wf(top=["my favourite extremely long descriptive lora name rank 105 bf16.safetensors"]),
            "SpacedStray.json": wf(top=["vae/my favourite extremely long descriptive lora name rank 105 bf16.safetensors"]),
            "Stray.json": wf(top=["vae/foo_vae.safetensors"]),
        })
        errors, warnings = vm.check_coverage(reg, wdir)

    check(not any("Note.json" in e for e in errors),
          "prose ending in a model URL must not be read as a filename")
    check(not any("Url.json" in e for e in errors),
          "a bare model URL must not be read as a filename")
    check(not any("Note.json" in w or "Url.json" in w for w in warnings),
          "skipped prose must not resurface as a user-supplied warning")
    check(not any("Spaced.json" in e for e in errors),
          "a filename containing spaces resolves cleanly against its registry entry")
    check(any("SpacedStray.json" in e for e in errors),
          "a spaced filename with a wrong prefix must still ERROR, which is the only "
          "observable proof it was COLLECTED rather than silently skipped. It is also "
          "long (79 chars with the prefix) and contains a slash, so this one case "
          "guards against a whitespace rule, a slash rule and any length cutoff. The "
          "family's longest real reference is 75 chars, so a len<64 heuristic would "
          "silently drop live ltx2 references")
    check(any("Stray.json" in e for e in errors),
          "a genuinely wrong folder prefix must still be a hard error")
    print("ok: prose and URLs skipped; spaced filenames still checked")


# --- private HF repos: the optional bearer token ----------------------------

@contextlib.contextmanager
def hf_token_env(value):
    """Set (or clear) HF_TOKEN for the block. The developer's own shell may
    export one, so every token test pins it rather than inheriting it."""
    before = os.environ.get("HF_TOKEN")
    if value is None:
        os.environ.pop("HF_TOKEN", None)
    else:
        os.environ["HF_TOKEN"] = value
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = before


class FakeResponse:
    """The little of urlopen's return value that http_request touches."""

    def __init__(self, url, status=200, headers=None, body=b"[]"):
        self.url, self.status, self.headers, self._body = url, status, headers or {}, body

    def read(self, n=-1):
        return self._body[:n] if n and n > 0 else self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def sent_request(url, range_first_byte=False):
    """Drive the REAL http_request with urlopen stubbed, and hand back the
    urllib Request it built, so the tests assert on actual outgoing headers."""
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return FakeResponse(req.full_url)

    real = vm.urllib.request.urlopen
    vm.urllib.request.urlopen = fake_urlopen
    try:
        REAL_HTTP_REQUEST(url, range_first_byte)
    finally:
        vm.urllib.request.urlopen = real
    return seen[0]


HF_TREE_URL = "https://huggingface.co/api/models/org/private/tree/main/vae"


def test_bearer_header_present_when_hf_token_set():
    """A private HF repo answers the tree listing only to an authenticated
    caller, so the client-template CI needs the header. Acceptance criterion:
    'validate_models.py resolves private HF repos when HF_TOKEN is present'."""
    with hf_token_env("hf_liveSECRETtokenvalue"):
        req = sent_request(HF_TREE_URL)
    check(req.get_header("Authorization") == "Bearer hf_liveSECRETtokenvalue",
          "token: HF_TOKEN set -> Authorization: Bearer <token> on the HF request")
    check(req.get_header("User-agent") == vm.UA["User-Agent"],
          "token: the User-Agent is still sent alongside it")


def test_no_bearer_header_without_hf_token():
    """The public four run with no token and must keep behaving EXACTLY as they
    do today; an empty-string HF_TOKEN is 'no token', not 'Bearer '."""
    with hf_token_env(None):
        req = sent_request(HF_TREE_URL)
    check(req.get_header("Authorization") is None,
          "token: HF_TOKEN unset -> no Authorization header at all")
    with hf_token_env(""):
        req = sent_request(HF_TREE_URL)
    check(req.get_header("Authorization") is None,
          "token: HF_TOKEN='' is treated as absent, never sent as 'Bearer '")


def test_bearer_header_never_leaves_huggingface():
    """http_request is also the presigned R2/S3 and Google Drive path. Attaching
    the customer's HF token to those requests would hand a live credential to a
    third-party host, so the header is scoped to huggingface.co over TLS."""
    with hf_token_env("hf_liveSECRETtokenvalue"):
        r2 = sent_request("https://r2.example.com/private/lora.safetensors?X-Amz-Signature=x",
                          range_first_byte=True)
        lookalike = sent_request("https://huggingface.co.evil.example/api/models/a/b/tree/main")
        cdn = sent_request("https://cdn-lfs.huggingface.co/repos/aa/bb/model.safetensors")
    check(r2.get_header("Authorization") is None,
          "token: no bearer token on a presigned R2/S3 URL")
    check(lookalike.get_header("Authorization") is None,
          "token: 'huggingface.co.evil.example' is not huggingface.co")
    check(cdn.get_header("Authorization") == "Bearer hf_liveSECRETtokenvalue",
          "token: an HF subdomain (cdn-lfs) still gets it")


def test_private_repo_listing_failure_names_the_missing_token():
    """Today this is the flat '{name}: could not list {repo}: HTTP 401'. On a
    client repo that is every asset, every build, and it names the symptom
    instead of the cause."""
    reg = {"client.safetensors": hf_entry("client.safetensors", "loras",
                                          repo="org/private", rdir="loras")}
    fake = FakeNet()
    fake.add("https://huggingface.co/api/models/org/private/tree/main/loras",
             401, {}, b"")
    vm.http_request = fake
    with hf_token_env(None):
        errors, _ = vm.check_urls(reg)
    check(len(errors) == 1 and "HF_TOKEN" in errors[0] and "private" in errors[0].lower(),
          f"token: a failed listing with no HF_TOKEN says the repo may be private "
          f"and no token was supplied (got {errors})")
    check(all("org/private" in e for e in errors),
          "token: and still names the repo that could not be listed")

    fake = FakeNet()
    fake.add("https://huggingface.co/api/models/org/private/tree/main/loras",
             403, {}, b"")
    vm.http_request = fake
    with hf_token_env("hf_liveSECRETtokenvalue"):
        errors, _ = vm.check_urls(reg)
    check(len(errors) == 1 and "403" in errors[0]
          and "hf_liveSECRETtokenvalue" not in errors[0],
          f"token: with a token set the error blames the token's access, and never "
          f"prints the token (got {errors})")


def test_token_never_reaches_an_error_string():
    """CLAUDE.md section 3: a raw exception is a leak channel. These messages are
    printed into CircleCI output and, on a pod, into $NETWORK_VOLUME/comfyui.log."""
    secret = "hf_liveSECRETtokenvalue"

    def raiser(url, range_first_byte=False):
        raise RuntimeError(f"tls error sending Authorization: Bearer {secret} to {url}")

    vm.http_request = raiser
    with hf_token_env(secret):
        hf_errors, _ = vm.check_urls(
            {"a.safetensors": hf_entry("a.safetensors", "loras", repo="org/private")})
        r2_errors, _ = vm.check_urls(
            {"b.safetensors": {"url": "https://r2.example.com/b.safetensors?sig=1",
                               "subdir": "loras"}})
    check(hf_errors and not any(secret in e for e in hf_errors),
          f"token: an exception quoting the token is scrubbed on the HF path (got {hf_errors})")
    check(r2_errors and not any(secret in e for e in r2_errors),
          f"token: and on the non-HF path too (got {r2_errors})")
    check(vm.scrub_token("nothing secret here") == "nothing secret here",
          "token: scrub_token leaves an innocent string alone")


def main() -> int:
    tests = [
        test_subgraph_walk,
        test_widget_prefix,
        test_unshipped_loader_ref_is_an_error,
        test_placeholders_are_allowed,
        test_typo_placeholder_still_errors,
        test_extra_prompt_is_not_scanned,
        test_raw_text_scan_still_runs_outside_extra_prompt,
        test_gated_tree_api,
        test_renamed_repo_named_in_error,
        test_tree_pagination,
        test_ranged_get,
        test_size_sanity,
        test_dest_subdir_rejected,
        test_baked_skips_network,
        test_template_checks,
        test_template_schema_accepts_both_real_shapes,
        test_template_schema_accepts_entrypoint_keys,
        test_template_schema_accepts_comfy_extra_args,
        test_template_schema_rejects_unknown_top_level_key,
        test_template_schema_rejects_unknown_flag_key,
        test_template_schema_rejects_unknown_swap_group_and_profile_shape,
        test_template_schema_runs_in_the_gate,
        test_presigned_url_is_redacted_in_errors,
        test_allowlists_suppress_warnings,
        test_offline_and_exit_codes,
        test_prose_is_not_a_filename,
        test_bearer_header_present_when_hf_token_set,
        test_no_bearer_header_without_hf_token,
        test_bearer_header_never_leaves_huggingface,
        test_private_repo_listing_failure_names_the_missing_token,
        test_token_never_reaches_an_error_string,
    ]
    for t in tests:
        t()
    failed = [label for ok, label in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for label in failed:
            print(f"FAILED: {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
