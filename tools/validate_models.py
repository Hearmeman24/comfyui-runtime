#!/usr/bin/env python3
"""CI-side model registry / workflow validator for the comfyui-runtime family.

Reconciles the three diverged template copies onto ltx2's superset
(comfyui-ltx2/tools/validate_models.py), folding in wan's allowlists and
user-supplied warning semantics (comfyui-wan/tools/validate_models.py),
minimax's copy (a strict subset of wan's), and qwen's baked-entry skip plus
flag-to-workflow existence check (.github/workflows/model_validator.py).

Offline checks (always run):
  1. Workflow JSON validity: every <workflows>/**/*.json must parse.
  2. Coverage: every model file a workflow references, in top-level nodes AND
     in definitions.subgraphs[].nodes, is resolved against the registry.
       - basename unknown to the registry (and not in template.json's
         auto_download / image_baked lists): user-supplied WARNING, never a
         failure; the customer provides it via CHECKPOINT_IDS/LORAS_IDS.
       - basename known but the widget value's folder prefix does not match
         the registry subdir: HARD ERROR. ComfyUI resolves a widget value
         relative to the category root, so "vae/foo.safetensors" for a model
         in models/vae/ resolves as models/vae/vae/foo.safetensors and shows
         a paying customer a missing model while the file is on the volume.
  3. Registry schema: url and subdir are required; dest_subdir is REJECTED
     (CONTRACTS.md section 4: dead field, no shims).
  4. template.json (when --template is given): flag "workflows" and "copy"
     entries must exist ("folders" that are missing only warn, matching the
     provisioner, comfyui-wan/src/workflow_provisioner.py:79-81); every
     swap-group profile filename must be a registry key (CONTRACTS.md 5a).

Network checks (skipped with --offline). Never a bare HEAD:
  5. Registry existence.
       - HF resolve URLs (gated or not) are checked against the HF model API
         tree listing (https://huggingface.co/api/models/<repo>/tree/<rev>/<dir>),
         which is public even for gated repos. An unauthenticated HEAD on a
         gated repo answers 401 whether or not the file exists, so it can
         never detect a deleted file; that blind spot let a renamed IC-LoRA
         (LipDub -> DubIt) sit broken while CI stayed green. Repo renames are
         followed and named in the error.
       - any other URL gets a ranged GET (Range: bytes=0-0, expecting 206
         with Content-Range). Presigned R2/S3 signatures are method-scoped,
         so a HEAD answers 403 on a GET-signed URL.
     A registry URL that does not resolve is a HARD ERROR (exit non-zero).
     Entries with "baked": true are skipped (the file ships in the image).
  6. Size sanity: a remote size under 10 MB is the tell for a model whose
     weights live in a sidecar no workflow references (the real case: a
     420 KB ONNX graph whose 2.5 GB _data.bin is invisible to a parse-derived
     list). WARNING, unless the entry's own min_size_mb vouches for the size
     or another registry entry declares the sidecar via auto_include_with.
     A remote size BELOW the entry's own min_size_mb is a HARD ERROR: the
     downloader would delete and refetch the file on every boot.

Stdlib only. Usage:

    python3 tools/validate_models.py \
        --registry  <template>/src/models_registry.json \
        --workflows <template>/workflows \
        [--template <template>/template.json] [--offline]
"""
from __future__ import annotations  # PEP 604 syntax under python < 3.10
import argparse
import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

TIMEOUT = 30
UA = {"User-Agent": "comfyui-runtime-validator/1.0"}
MB = 1024 * 1024
# The family's default "looks complete" floor (CONTRACTS.md section 1).
SMALL_FLOOR_MB = 10

# Union of wan's extensions (comfyui-wan/tools/validate_models.py:30) and
# qwen's pt/gguf (.github/workflows/model_validator.py:22), per CONTRACTS.md
# section 3 step 2.
MODEL_EXTS = (".safetensors", ".bin", ".onnx", ".pth", ".ckpt", ".pt", ".gguf")
MODEL_PAT = re.compile(r'"([^"]+\.(?:safetensors|bin|onnx|pth|ckpt|pt|gguf))"')

# The only unregistered basenames a shipped workflow may name. Everything else
# a loader points at must be in the registry, or the template promises a model
# it never delivers (E13). Deliberately an explicit set, not a `Your_*` prefix
# match: a typo'd placeholder should fail loudly, and adding a fourth name
# should be a decision someone makes on purpose rather than a filename that
# happens to fit a pattern.
PLACEHOLDERS = frozenset({
    "Your_Character_LoRA_Here.safetensors",
    "Your_LoRA_Here_HIGH_NOISE.safetensors",
    "Your_LoRA_Here_LOW_NOISE.safetensors",
})
# https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path/in/repo>
HF_RESOLVE_RE = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<file>.+?)(?:\?.*)?$")


# ---------------------------------------------------------------- network ---

def http_request(url: str, range_first_byte: bool = False):
    """The single network seam; the self-test replaces this. GET only, never
    HEAD. Returns (status, headers, body, final_url). HTTP error statuses are
    returned, not raised; transport errors raise."""
    req = urllib.request.Request(url, headers=dict(UA))
    if range_first_byte:
        req.add_header("Range", "bytes=0-0")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = r.read(1) if range_first_byte else r.read()
            return r.status, dict(r.headers), body, r.url
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), b"", url


def _next_link(headers: dict) -> str | None:
    """Cursor pagination of the HF tree API (Link: <url>; rel="next")."""
    link = headers.get("Link") or headers.get("link") or ""
    for part in link.split(","):
        segs = part.split(";")
        if len(segs) >= 2 and 'rel="next"' in segs[1]:
            return segs[0].strip().strip("<>")
    return None


def hf_tree_listing(repo: str, rev: str, dirpath: str) -> tuple[str, dict]:
    """(landed_repo, {path_in_repo: size}) from the public HF model API tree
    listing, which is public even for gated repos. urllib follows the 307 a
    renamed repo issues; recover where we landed so the error can name it."""
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}"
    if dirpath:
        url += f"/{dirpath}"
    landed, files = repo, {}
    while url:
        status, headers, body, final_url = http_request(url)
        if status != 200:
            raise RuntimeError(f"HTTP {status} listing {url}")
        m = re.search(r"/api/models/([^/]+/[^/]+)/tree/", final_url)
        if m:
            landed = m.group(1)
        for item in json.loads(body):
            if isinstance(item, dict) and item.get("type") == "file":
                files[item.get("path", "")] = item.get("size")
        url = _next_link(headers)
    return landed, files


# ------------------------------------------------------------------ checks --

def check_workflow_json(workflows_dir: Path) -> list[str]:
    errors = []
    for wf in sorted(Path(workflows_dir).rglob("*.json")):
        try:
            json.loads(wf.read_text())
        except Exception as e:
            errors.append(f"invalid JSON: {wf.relative_to(workflows_dir)}: {e}")
    return errors


def check_registry_schema(registry: dict) -> list[str]:
    errors = []
    for name, entry in registry.items():
        if not isinstance(entry, dict):
            errors.append(f"{name}: registry entry must be a JSON object")
            continue
        if "dest_subdir" in entry:
            errors.append(f"{name}: 'dest_subdir' is dead (CONTRACTS.md section 4, no shims); "
                          f"use 'subdir' relative to the models root, no 'models/' prefix")
        for field in ("url", "subdir"):
            if not entry.get(field):
                errors.append(f"{name}: missing required field '{field}'")
    return errors


def _looks_like_a_filename(val: str) -> bool:
    """A widget value that merely ENDS in a model extension is not necessarily a
    model reference. A MarkdownNote whose prose closes with a download link ends
    in ".safetensors" too, and the coverage check then derives a "folder prefix"
    from a paragraph and errors on it (a real wan workflow does exactly this).

    Two things a filename is never: multi-line, or a URL. Reject on those.

    Deliberately NOT rejecting on whitespace generally: ComfyUI filenames may
    contain spaces, and skipping those would turn a loud false positive into a
    silent false negative, which is the worse failure.
    """
    return "\n" not in val and "\r" not in val and "://" not in val


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)


def workflow_widget_refs(doc: dict) -> set[str]:
    """Model-shaped widget values from top-level nodes AND subgraph
    definitions. Loaders increasingly live inside definitions.subgraphs
    (the LTX-2.5 workflows keep every loader in one), so walking only
    doc["nodes"] silently checks nothing there."""
    groups = [doc.get("nodes") or []]
    groups += [sg.get("nodes") or [] for sg in
               ((doc.get("definitions") or {}).get("subgraphs") or [])]
    refs = set()
    for nodes in groups:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            for val in _iter_strings(node.get("widgets_values") or []):
                if val.lower().endswith(MODEL_EXTS) and _looks_like_a_filename(val):
                    refs.add(val)
    return refs


def scannable_text(doc: dict, fallback: str) -> str:
    """The file's text with `extra.prompt` removed.

    `extra.prompt` is an API-format snapshot the frontend stashes beside the
    graph: the flat {"1": {"class_type", "inputs"}} form that gets POSTed to
    /prompt. ComfyUI executes doc["nodes"], never this, and the snapshot goes
    stale the moment the graph is rewritten without re-saving.

    Comfy-Org ships exactly that in its own LTX-2.5 templates: both carry a
    26-node snapshot of a hand-wired pipeline (CheckpointLoaderSimple,
    LTXVGemmaCLIPModelLoader, gemma-3-12b) under a live graph that is now a
    single subgraph node. Scanning it reported three models as missing on a
    pod where nothing was missing. Warnings people learn to ignore are worse
    than no warnings, and this one is upstream debris we cannot fix at source.
    """
    if not isinstance(doc, dict):
        return fallback
    extra = doc.get("extra")
    if not isinstance(extra, dict) or "prompt" not in extra:
        return fallback
    pruned = {**doc, "extra": {k: v for k, v in extra.items() if k != "prompt"}}
    try:
        return json.dumps(pruned)
    except (TypeError, ValueError):
        return fallback  # unserialisable doc: scan everything rather than nothing


def check_coverage(registry: dict, workflows_dir: Path,
                   allow: frozenset = frozenset()) -> tuple[list[str], list[str]]:
    """(errors, warnings).

    A loader widget naming a file the template does not ship is an ERROR
    unless it is a PLACEHOLDER: that is the E13 gate, and it is what stops a
    personal LoRA leaking into a shipped workflow again. A known basename
    referenced with the wrong folder prefix is an ERROR too.

    The raw-text scan stays warning-level. It reads note prose and
    properties.models, where a false positive is cheap and a hard failure
    would be wrong: properties.models is a fetch-if-missing hint, not what
    the loader loads.
    """
    errors, warnings = [], []
    warned: set[str] = set()
    for wf in sorted(Path(workflows_dir).rglob("*.json")):
        try:
            raw = wf.read_text()
            doc = json.loads(raw)
        except Exception:
            continue  # reported by check_workflow_json
        text = scannable_text(doc, raw)
        rel = wf.relative_to(workflows_dir)
        widget_refs = workflow_widget_refs(doc) if isinstance(doc, dict) else set()
        for ref in sorted(widget_refs):
            name = PurePosixPath(ref).name
            if name in PLACEHOLDERS:
                continue  # deliberate empty slot the customer fills in
            if name not in registry:
                if name not in allow:
                    errors.append(
                        f"{rel}: loader references '{name}', which this template does "
                        f"not ship. Add it to models_registry.json, or blank the slot "
                        f"with a placeholder ({sorted(PLACEHOLDERS)[0]}).")
                continue
            subdir = registry[name].get("subdir") if isinstance(registry[name], dict) else None
            if not subdir:
                continue  # already a schema error
            # ComfyUI resolves a widget value relative to the category root,
            # so only a genuine subfolder belongs in the prefix: registry
            # subdir "vae" -> "", "loras/ltx2" -> "ltx2".
            want = "/".join(subdir.split("/")[1:])
            got = str(PurePosixPath(ref).parent) if "/" in ref else ""
            if got == ".":
                got = ""
            if got != want:
                shown = f"'{want}/{name}'" if want else f"'{name}'"
                errors.append(
                    f"{rel}: references '{ref}' with the wrong folder prefix; "
                    f"ComfyUI resolves it under models/{subdir}/. Use {shown}.")
        # Raw-text scan for refs outside widgets_values (properties.models,
        # note prose), warning-level only: comfyui-wan/tools/validate_models.py:30,48.
        widget_names = {PurePosixPath(r).name for r in widget_refs}
        for m in MODEL_PAT.findall(text):
            name = PurePosixPath(m).name
            if (name in registry or name in allow or name in widget_names
                    or name in warned or name in PLACEHOLDERS):
                continue
            warned.add(name)
            warnings.append(f"user-supplied (not in registry): {name}  [{rel}]")
    return errors, warnings


def check_template(template: dict, registry: dict,
                   workflows_dir: Path) -> tuple[list[str], list[str]]:
    """(errors, warnings) for template.json cross-references."""
    errors, warnings = [], []
    workflows_dir = Path(workflows_dir)
    for flag, cfg in (template.get("flags") or {}).items():
        if not isinstance(cfg, dict):
            errors.append(f"template.json: flag {flag} must map to an object")
            continue
        # qwen's check: a flag naming a missing workflow file is an error
        # (.github/workflows/model_validator.py:50-52).
        for wf in cfg.get("workflows") or []:
            if not (workflows_dir / wf).is_file():
                errors.append(f"template.json: flag {flag} references missing workflow '{wf}'")
        # A missing folder is a warning only (wan workflow_provisioner.py:79-81).
        for folder in cfg.get("folders") or []:
            if not (workflows_dir / folder).is_dir():
                warnings.append(f"template.json: flag {flag} names missing folder '{folder}'")
        # Registry-mode copy entries are paths relative to workflows-src;
        # "." means the top level and always exists (CONTRACTS.md section 3).
        for cp in cfg.get("copy") or []:
            if cp != "." and not (workflows_dir / cp).exists():
                errors.append(f"template.json: flag {flag} copy entry '{cp}' does not exist")
    # A swap-group profile filename missing from the registry is a config
    # error at provision time (exit 2, CONTRACTS.md section 5a); catch it in CI.
    for group in template.get("swap_groups") or []:
        env = group.get("env", "?")
        for pname, profile in (group.get("profiles") or {}).items():
            for role, fname in (profile or {}).items():
                if fname not in registry:
                    errors.append(f"template.json: swap group '{env}' profile '{pname}' "
                                  f"role '{role}': '{fname}' is not in the registry")
    return errors, warnings


def _size_check(name: str, entry: dict, size, registry: dict):
    """(error, warning) for the remote size of one entry."""
    if size is None:
        return None, None
    declared = entry.get("min_size_mb")
    if declared is not None:
        if size < float(declared) * MB:
            return (f"{name}: remote file is {size / MB:.2f} MB, below its own min_size_mb "
                    f"({declared}); the downloader would delete and refetch it on every boot"), None
        return None, None  # the entry vouches for its own size
    if size >= SMALL_FLOOR_MB * MB:
        return None, None
    if any(isinstance(e, dict) and e.get("auto_include_with") == name
           for e in registry.values()):
        return None, None  # the weights sidecar is declared; a small graph file is expected
    return None, (f"{name}: implausibly small ({size / MB:.2f} MB); if its weights live in a "
                  f"sidecar (e.g. an ONNX _data.bin) declare it via auto_include_with, and set "
                  f"min_size_mb (the default {SMALL_FLOOR_MB} MB floor would refetch this file "
                  f"on every boot)")


def check_url(name: str, entry: dict, registry: dict):
    """(error, warning) for one registry entry's URL. Never issues a HEAD."""
    url = entry["url"]
    m = HF_RESOLVE_RE.match(url)
    if m:
        repo, rev, fpath = m["repo"], m["rev"], m["file"]
        dirpath = str(PurePosixPath(fpath).parent)
        if dirpath == ".":
            dirpath = ""
        try:
            landed, files = hf_tree_listing(repo, rev, dirpath)
        except Exception as e:  # noqa: BLE001
            return f"{name}: could not list {repo}: {e}", None
        if fpath not in files:
            if landed.lower() != repo.lower():
                return (f"{name}: '{fpath}' not in {landed} - {repo} was renamed to {landed} "
                        f"and the file did not come across under this name"), None
            return f"{name}: '{fpath}' is not in {repo} (removed or renamed)", None
        return _size_check(name, entry, files[fpath], registry)
    # Non-HF (presigned R2/S3, GitHub raw, Google Drive): ranged GET. A HEAD
    # returns 403 on a GET-signed URL because the signature is method-scoped.
    try:
        status, headers, _, _ = http_request(url, range_first_byte=True)
    except Exception as e:  # noqa: BLE001
        return f"{name}: {e} - {url}", None
    if status == 206:
        cr = re.match(r"bytes\s+\d+-\d+/(\d+)", headers.get("Content-Range", ""))
        size = int(cr.group(1)) if cr else None
    elif status == 200:  # server ignored the Range header; the file exists
        cl = headers.get("Content-Length", "")
        size = int(cl) if cl.isdigit() else None
    else:
        return f"{name}: HTTP {status} on ranged GET - {url}", None
    return _size_check(name, entry, size, registry)


def check_urls(registry: dict) -> tuple[list[str], list[str]]:
    jobs = [(n, e) for n, e in registry.items()
            if isinstance(e, dict) and e.get("url") and not e.get("baked")]
    errors, warnings = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda j: check_url(j[0], j[1], registry), jobs))
    for err, warn in results:
        if err:
            errors.append(err)
        if warn:
            warnings.append(warn)
    return errors, warnings


# -------------------------------------------------------------------- main --

def run(registry_path, workflows_dir, template_path=None, offline=False) -> int:
    registry_path, workflows_dir = Path(registry_path), Path(workflows_dir)
    try:
        registry = json.loads(registry_path.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"❌ cannot read registry {registry_path}: {e}")
        return 1
    if not isinstance(registry, dict):
        print(f"❌ {registry_path}: registry must be one flat JSON object")
        return 1
    if not workflows_dir.is_dir():
        print(f"❌ workflows dir not found: {workflows_dir}")
        return 1

    template, allow = None, frozenset()
    if template_path:
        try:
            template = json.loads(Path(template_path).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"❌ cannot read template {template_path}: {e}")
            return 1
        allow = frozenset((template.get("auto_download") or []) +
                          (template.get("image_baked") or []))

    errors = check_registry_schema(registry)
    errors += check_workflow_json(workflows_dir)
    cov_errors, warnings = check_coverage(registry, workflows_dir, allow)
    errors += cov_errors
    if template is not None:
        tpl_errors, tpl_warnings = check_template(template, registry, workflows_dir)
        errors += tpl_errors
        warnings += tpl_warnings

    if offline:
        print("⏭️  --offline: skipping upstream existence and size checks.")
    else:
        n = sum(1 for e in registry.values()
                if isinstance(e, dict) and e.get("url") and not e.get("baked"))
        gated = sum(1 for e in registry.values()
                    if isinstance(e, dict) and e.get("gated") and not e.get("baked"))
        print(f"🔎 verifying {n} registry files exist upstream ({gated} gated)...")
        url_errors, url_warnings = check_urls(registry)
        errors += url_errors
        warnings += url_warnings

    for w in warnings:
        print(f"⚠️  {w}")
    for e in errors:
        print(f"❌ {e}")
    if errors:
        print(f"💥 {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    print(f"✅ validation passed: {len(registry)} registry entries, {len(warnings)} warning(s)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a template's model registry, "
                                             "workflows and template.json.")
    ap.add_argument("--registry", required=True, help="path to models_registry.json")
    ap.add_argument("--workflows", required=True, help="path to the workflows dir")
    ap.add_argument("--template", default=None, help="path to template.json (optional)")
    ap.add_argument("--offline", action="store_true",
                    help="skip the upstream existence and size checks")
    args = ap.parse_args(argv)
    return run(args.registry, args.workflows, args.template, args.offline)


if __name__ == "__main__":
    raise SystemExit(main())
