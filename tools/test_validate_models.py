#!/usr/bin/env python3
"""Offline self-test for tools/validate_models.py.

Run: python3 tools/test_validate_models.py

No network is ever touched: the module's single seam (http_request) is
replaced with a fake that serves canned HF tree listings and ranged-GET
responses, and records every call so the tests can assert what was (and
was not) requested. Stdlib only, no pytest.
"""
from __future__ import annotations  # PEP 604 syntax under python < 3.10
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_models as vm  # noqa: E402

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
    check(any("sub_mystery.safetensors" in w for w in warnings),
          "gap 1: user-supplied warning is found inside a subgraph-only workflow")
    check(not any("sub_mystery" in e for e in errors),
          "severity: a basename absent from the registry is a warning, not an error")


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
        "my favourite lora.safetensors": hf_entry("my favourite lora.safetensors", "loras"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        wdir = make_workflows(tmp, {
            "Note.json": wf(top=[prose]),
            "Url.json": wf(top=[bare_url]),
            "Spaced.json": wf(top=["my favourite lora.safetensors"]),
            "SpacedStray.json": wf(top=["vae/my favourite lora.safetensors"]),
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
          "observable proof it was COLLECTED rather than silently skipped")
    check(any("Stray.json" in e for e in errors),
          "a genuinely wrong folder prefix must still be a hard error")
    print("ok: prose and URLs skipped; spaced filenames still checked")


def main() -> int:
    tests = [
        test_subgraph_walk,
        test_widget_prefix,
        test_gated_tree_api,
        test_renamed_repo_named_in_error,
        test_tree_pagination,
        test_ranged_get,
        test_size_sanity,
        test_dest_subdir_rejected,
        test_baked_skips_network,
        test_template_checks,
        test_allowlists_suppress_warnings,
        test_offline_and_exit_codes,
        test_prose_is_not_a_filename,
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
