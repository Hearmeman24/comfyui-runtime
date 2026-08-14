#!/usr/bin/env python3
"""Unified workflow/model provisioner (CONTRACTS.md section 3).

One tool, two modes, replacing:
  - comfyui-wan/src/workflow_provisioner.py       (walk)
  - comfyui-minimax/src/workflow_provisioner.py   (walk + quant swap)
  - comfyui-qwen-image/src/provision_models.py    (walk, explicit lists +
                                                   precision swap)
  - comfyui-ltx2/src/build_manifest.py            (registry)

Flag state and variant/quant/precision choices are read from the process
environment, mapped through template.json. There is no --flag and no --quant
argument: start.sh is shared and cannot know per-template flag names.

Source of truth in walk mode is the workflows tree itself. A model reference
with no registry entry surfaces as a user-supplied warning, never a hard
error (the CI validator enforces registry coverage at merge time).

Output: the download manifest for hf_download_manager.py, one
`<url>\\t<abs_dest>[\\t<min_size_mb>]` line per model (CONTRACTS.md section 1),
plus the selected workflow JSONs copied into --workflows-dst preserving each
file's path relative to --workflows-src, retargeted onto the selected swap
profiles in BOTH the loader widgets and each node's `properties.models` (what
the ComfyUI frontend's "Missing Models" dialog reads), at the top level AND
inside `definitions.subgraphs[].nodes`.

Run the self-check with: python3 provisioner.py --selftest
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# Union of wan's and qwen's extension sets
# (comfyui-wan/src/workflow_provisioner.py:44 adds safetensors|bin|onnx|pth|
# ckpt; comfyui-qwen-image/src/provision_models.py:30 adds pt|gguf).
MODEL_PAT = re.compile(r'"([^"]+\.(?:safetensors|bin|onnx|pth|pt|ckpt|gguf))"')

# 10 MB "looks complete" floor (comfyui-wan/src/workflow_provisioner.py:106,
# comfyui-ltx2/src/build_manifest.py:39).
MIN_OK_SIZE = 10 * 1024 * 1024

# Opt-in truthiness, deliberately wider than wan/minimax's bare "true"
# (comfyui-qwen-image/src/provision_models.py:46-50; CONTRACTS.md section 3).
TRUTHY = {"1", "true", "yes", "on"}


class ConfigError(Exception):
    """template.json / registry misconfiguration: exit 2."""


def flag_enabled(env, name: str, default: bool) -> bool:
    """CONTRACTS.md section 3 flag truthiness, both modes."""
    raw = (env.get(name) or "").strip().lower()
    if default:
        # Opt-out: only a literal "false" disables, so a typo cannot
        # silently strip the default set (ltx2 build_manifest.py:59-68).
        return raw != "false"
    return raw in TRUTHY


def select(registry: dict, env, lightweight_fp8: bool) -> dict:
    """Filter the registry by flag gating + fp8/full variant choice.

    Ported from comfyui-ltx2/src/build_manifest.py:42-81. Opt-in flag
    truthiness is widened to TRUTHY per CONTRACTS.md section 3; on_by_default
    (literal "false" opts out) and disable_flag (exactly "true" drops) keep
    ltx2's semantics verbatim.

    - "flag": <env>         opt-IN, unless "on_by_default" makes it opt-OUT.
    - "disable_flag": <env> opt-OUT: kept unless that env is exactly "true".
    - "variant": full|fp8   among variant-tagged entries keep one side.
    - "gated": true         HF-gated repo, queued only when HF_TOKEN is set
                            and non-blank; without a token the download would
                            403, so drop it (fail-open).
    """
    has_token = bool((env.get("HF_TOKEN") or "").strip())

    def flag_ok(e: dict) -> bool:
        flag = e.get("flag")
        if not flag:
            return True
        if e.get("on_by_default"):
            return (env.get(flag) or "").strip().lower() != "false"
        return flag_enabled(env, flag, default=False)

    kept = {
        n: e for n, e in registry.items()
        if flag_ok(e)
        and env.get(e.get("disable_flag") or "") != "true"
        and (not e.get("gated") or has_token)
    }
    want = "fp8" if lightweight_fp8 else "full"
    return {
        n: e for n, e in kept.items()
        if not e.get("variant") or e["variant"] == want
    }


def resolve_profile_key(group: dict, env) -> str:
    """Pick a swap group's profile from its env var. Unknown values warn and
    fall back to the default: a customer typo must not kill a boot
    (qwen resolve_precision, provision_models.py:65-70)."""
    profiles = group["profiles"]
    raw = env.get(group["env"])
    if raw is None:
        return group["default"]
    if raw in profiles:
        return raw
    lowered = raw.strip().lower()
    if lowered in profiles:
        return lowered
    print(f"[provisioner] warning: unknown {group['env']}={raw!r}; "
          f"using default {group['default']!r}")
    return group["default"]


def build_swap_state(template: dict, env, enabled: set,
                     registry: dict) -> tuple:
    """Returns (swaps, managed, selected_files).

    Generalised from comfyui-minimax/src/workflow_provisioner.py:34-51,
    100-105,126-132 (CONTRACTS.md section 5a):
      - managed: every filename in every profile of every group; scan refs
        to these never queue directly (minimax :51,104-105).
      - swaps: P[role] -> S[role] for each profile P of each ACTIVE group
        (minimax :131-132).
      - selected_files: the active groups' selected-profile files, queued in
        walk mode regardless of scan results (minimax :100-101). A selected
        file missing from the registry is a config error, exit 2 (today a
        raw KeyError at minimax :101).
    """
    swaps: dict = {}
    managed: set = set()
    selected_files: list = []
    for group in template.get("swap_groups", []):
        for profile in group["profiles"].values():
            managed.update(profile.values())
        if not any(f in enabled for f in group.get("flags", [])):
            continue
        if group["default"] not in group["profiles"]:
            raise ConfigError(
                f"swap group {group['env']!r}: default "
                f"{group['default']!r} is not a profile")
        sel = group["profiles"][resolve_profile_key(group, env)]
        for name in sel.values():
            if name not in registry:
                raise ConfigError(
                    f"swap profile file not in registry: {name}")
        selected_files.extend(sel.values())
        for profile in group["profiles"].values():
            for role, fname in profile.items():
                if role in sel and fname != sel[role]:
                    swaps[fname] = sel[role]
    return swaps, managed, selected_files


def rewrite_doc(doc: dict, swaps: dict, registry: dict) -> None:
    """Retarget a workflow onto the selected profiles, in place.

    Ported from comfyui-minimax/src/workflow_provisioner.py:141-160. Both
    the top-level nodes and definitions.subgraphs[].nodes are walked: a
    subgraph INSTANCE node carries its own promoted widget values separate
    from the definition's inner loaders. Only exact filename widget values
    are swapped, so note prose survives.
    """
    groups = [doc.get("nodes", [])] + [
        sg.get("nodes", [])
        for sg in doc.get("definitions", {}).get("subgraphs", [])
    ]
    for group in groups:
        for n in group:
            wv = n.get("widgets_values")
            if isinstance(wv, list):
                n["widgets_values"] = [
                    swaps.get(v, v) if isinstance(v, str) else v for v in wv
                ]
            # The frontend's "Missing Models" dialog reads properties.models,
            # not the loader widget. Leave it on the variant we didn't
            # download and the customer is told a file is missing and handed
            # a button that saves it to their own PC.
            for m in (n.get("properties") or {}).get("models") or []:
                new = swaps.get(m.get("name"))
                if new:
                    m["name"] = new
                    m["url"] = registry[new]["url"]


def gather_walk(flags_map: dict, enabled: list, src_root: Path) -> list:
    """Walk-mode workflow gathering: recursive *.json per flag folder
    (wan :77-85) or qwen's explicit file lists (provision_models.py:92-102).
    A missing folder/file is a warning, never a failure (wan :79-81)."""
    files: dict = {}
    for name in enabled:
        cfg = flags_map[name]
        for folder in cfg.get("folders", []):
            d = src_root / folder
            if not d.is_dir():
                print(f"[provisioner] warning: workflow folder missing: {d}")
                continue
            for wf in sorted(d.rglob("*.json")):
                files[wf] = wf
        for rel in cfg.get("workflows", []):
            p = src_root / rel
            if not p.is_file():
                print(f"[provisioner] warning: workflow file missing: {p}")
                continue
            files[p] = p
    return list(files.values())


def gather_registry(flags_map: dict, enabled: list, src_root: Path) -> list:
    """Registry-mode copy lists (CONTRACTS.md section 3, replacing
    comfyui-ltx2/src/start.sh:378-403): "." means top-level *.json only,
    non-recursive; a directory means all *.json under it recursively."""
    files: dict = {}
    for name in enabled:
        for entry in flags_map[name].get("copy", []):
            if entry == ".":
                found = sorted(src_root.glob("*.json"))
            else:
                p = src_root / entry
                if p.is_dir():
                    found = sorted(p.rglob("*.json"))
                elif p.is_file():
                    found = [p]
                else:
                    print(f"[provisioner] warning: copy entry missing: {p}")
                    continue
            for wf in found:
                files[wf] = wf
    return list(files.values())


def copy_workflows(files: list, src_root: Path, dst_root: Path,
                   swaps: dict, registry: dict) -> list:
    """Copy preserving each file's path relative to --workflows-src
    (wan :118-125). With a non-empty swap map the copy is a JSON rewrite
    (minimax :134-160); otherwise bytes verbatim (wan :124).
    Returns [(dest, post-rewrite content)]."""
    out_docs = []
    dst_root.mkdir(parents=True, exist_ok=True)
    for wf in files:
        rel = wf.relative_to(src_root)
        out = dst_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = wf.read_text()
        if swaps:
            doc = json.loads(text)
            if isinstance(doc, dict):
                rewrite_doc(doc, swaps, registry)
                text = json.dumps(doc, indent=2, ensure_ascii=False)
            out.write_text(text)
        else:
            shutil.copy2(wf, out)
        out_docs.append((out, text))
    return out_docs


def build(queued: dict, models_root: Path) -> tuple:
    """Skip-existing floor + manifest lines (ltx2 build_manifest.py:84-97).
    min_size_mb overrides the flat floor and rides along as the manifest's
    third field so the downloader verifies against the same number
    (CONTRACTS.md section 1)."""
    lines, skipped = [], []
    for name in sorted(queued):
        entry = queued[name]
        dest = models_root / entry["subdir"] / name
        floor = int(entry.get("min_size_mb", 0) * 1024 * 1024) or MIN_OK_SIZE
        if dest.is_file() and dest.stat().st_size >= floor:
            skipped.append(name)
            continue
        line = f"{entry['url']}\t{dest}"
        if "min_size_mb" in entry:
            line += f"\t{entry['min_size_mb']}"
        lines.append(line)
    return lines, skipped


def write_status(env, payload) -> None:
    """Optional boot-report feed (EXECUTION.md item N1): when
    PROVISION_STATUS_FILE is set in the environment, write a machine-readable
    summary of what this run decided, for boot_report.py. Never fatal; the
    deployment report renders "unknown" rows without it."""
    path = env.get("PROVISION_STATUS_FILE")
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(payload, indent=1))
    except OSError as e:
        print(f"[provisioner] could not write status file: {e}")


def provision(args, env) -> int:
    try:
        template = json.loads(Path(args.template).read_text())
        registry = json.loads(Path(args.registry).read_text())
        if not isinstance(template, dict) or not isinstance(registry, dict):
            raise ConfigError("template/registry must be JSON objects")
    except (OSError, json.JSONDecodeError, ConfigError) as e:
        print(f"[provisioner] config error: {e}", file=sys.stderr)
        return 2
    mode = template.get("provisioning_mode")
    if mode not in ("walk", "registry"):
        print(f"[provisioner] config error: unknown provisioning_mode: "
              f"{mode!r}", file=sys.stderr)
        return 2

    src_root = Path(args.workflows_src)
    dst_root = Path(args.workflows_dst)
    models_root = Path(args.models_root)
    manifest = Path(args.manifest)

    flags_map = template.get("flags", {})
    # Retired flags: still accepted so a saved RunPod template keeps booting,
    # but they enable nothing and copy nothing. Announced only when the
    # customer actually has one set, so a clean pod stays quiet.
    deprecated = template.get("deprecated_flags", {}) or {}
    retired = frozenset(deprecated)
    for name in sorted(deprecated):
        if flag_enabled(env, name, False):
            print(f"[provisioner] {name} is RETIRED and ships nothing. "
                  f"{deprecated[name]}")
    flags_map = {n: c for n, c in flags_map.items() if n not in retired}
    enabled = [n for n, cfg in flags_map.items()
               if flag_enabled(env, n, bool(cfg.get("default")))]

    if mode == "walk" and not enabled:
        # (wan workflow_provisioner.py:63-66)
        print("[provisioner] no flags enabled: empty manifest, "
              "no workflows copied")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("")
        write_status(env, {"mode": mode, "enabled_flags": [],
                           "workflows_copied": [], "skipped": [],
                           "user_supplied": []})
        return 0

    try:
        swaps, managed, selected_files = build_swap_state(
            template, env, set(enabled), registry)
    except ConfigError as e:
        print(f"[provisioner] config error: {e}", file=sys.stderr)
        return 2

    if mode == "walk":
        files = gather_walk(flags_map, enabled, src_root)
    else:
        files = gather_registry(flags_map, enabled, src_root)
    out_docs = copy_workflows(files, src_root, dst_root, swaps, registry)

    # Scan pass over POST-rewrite content (wan :82-85, minimax :92-95).
    refs: set = set()
    for _, text in out_docs:
        refs.update(os.path.basename(m) for m in MODEL_PAT.findall(text))

    if mode == "registry":
        # Registry gating decides downloads; the scan below is informational
        # only, producing user-supplied warnings (CONTRACTS.md section 3).
        # The fp8/full download choice must resolve the env var exactly as
        # the workflow rewrite did, or the manifest pulls one variant while
        # the widgets load the other. When a swap group shares the variant
        # env, resolve_profile_key is the single source of truth; without
        # one there is no rewrite side, so a normalized compare suffices.
        variant_env = template.get("variant_env")
        # Guard on variant_env being set before matching: with it absent, a
        # group that is itself missing "env" would match None == None and
        # resolve_profile_key would KeyError, failing the whole provision on
        # a schema-invalid template that used to boot.
        group = next((g for g in template.get("swap_groups", [])
                      if variant_env and g.get("env") == variant_env), None)
        if group is not None:
            lightweight_fp8 = resolve_profile_key(group, env) == "true"
        else:
            lightweight_fp8 = bool(variant_env) and (
                env.get(variant_env) or "").strip().lower() == "true"
        queued = dict(select(registry, env, lightweight_fp8))
    else:
        # Swap-group queue rule: the selected profiles are pulled regardless
        # of scan results (minimax :100-101).
        queued = {b: registry[b] for b in selected_files}

    # Resolve scanned refs (wan :89-95; managed-set skip minimax :104-105).
    auto_download = set(template.get("auto_download", []))
    image_baked = set(template.get("image_baked", []))
    user_supplied: set = set()
    for b in refs:
        if b in auto_download or b in image_baked or b in managed:
            continue
        entry = registry.get(b)
        if entry is None:
            user_supplied.add(b)
        elif mode == "walk":
            queued[b] = entry

    # Per-flag extra_models: a missing name is an error line, not a failure
    # (qwen provision_models.py:119-127).
    for name in enabled:
        for b in flags_map[name].get("extra_models", []):
            entry = registry.get(b)
            if entry is None:
                print(f"[provisioner] error: {name}: extra_models entry "
                      f"not in registry: {b}")
            else:
                queued[b] = entry

    # Sidecars run after all gating (wan :97-102; CONTRACTS.md section 4).
    for b, entry in registry.items():
        trigger = entry.get("auto_include_with")
        if trigger and trigger in queued and b not in queued:
            queued[b] = entry
    # baked files are never queued (qwen :107-108; CONTRACTS.md section 4).
    queued = {b: e for b, e in queued.items() if not e.get("baked")}
    # Retired flags ship nothing, in BOTH modes. Filtering here rather than in
    # select() also closes the walk-mode path, where a retired entry could
    # otherwise still arrive as an auto_include_with sidecar of a live one.
    # The env value itself stays accepted: announced above, never an error.
    if retired:
        queued = {b: e for b, e in queued.items()
                  if e.get("flag") not in retired}

    lines, skipped = build(queued, models_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""))
    write_status(env, {
        "mode": mode,
        "enabled_flags": enabled,
        "workflows_copied": [str(dest.relative_to(dst_root))
                             for dest, _ in out_docs],
        "skipped": skipped,
        "user_supplied": sorted(user_supplied),
    })

    print(f"[provisioner] mode: {mode}  "
          f"flags: {','.join(enabled) or '(none)'}")
    print(f"[provisioner] workflows copied: {len(out_docs)}")
    print(f"[provisioner] models queued: {len(lines)} "
          f"({len(skipped)} already on disk, skipped)")
    if user_supplied:
        print(f"[provisioner] {len(user_supplied)} user-supplied file(s) "
              "referenced but not in registry - place them manually or via "
              "CHECKPOINT_IDS_TO_DOWNLOAD/LORAS_IDS_TO_DOWNLOAD:")
        for b in sorted(user_supplied):
            print(f"  - {b}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--workflows-src", required=True)
    ap.add_argument("--workflows-dst", required=True)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    try:
        return provision(args, os.environ)
    except Exception as e:  # noqa: BLE001 - CONTRACTS.md section 3, exit 1
        print(f"[provisioner] unexpected error: {e}", file=sys.stderr)
        return 1


def selftest() -> None:
    """select() gating + floor logic (ported with its coverage from
    comfyui-ltx2/src/build_manifest.py:100-207) + swap-state units."""
    import tempfile
    reg = {
        "big.safetensors": {"url": "https://h/big", "subdir": "loras/ltx2"},
        "missing.safetensors": {"url": "https://h/missing", "subdir": "vae"},
    }
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        big = root / "loras/ltx2/big.safetensors"
        big.parent.mkdir(parents=True)
        big.write_bytes(b"\0" * (MIN_OK_SIZE + 1))  # present + large -> skip
        lines, skipped = build(reg, root)
        assert skipped == ["big.safetensors"], skipped
        assert lines == [
            f"https://h/missing\t{root / 'vae/missing.safetensors'}"], lines
        big.write_bytes(b"\0" * 10)  # truncated -> must re-queue
        lines2, skipped2 = build(reg, root)
        assert skipped2 == [] and len(lines2) == 2, (skipped2, lines2)

    # flag gating + variant selection
    greg = {
        "always.safetensors": {"url": "https://h/a", "subdir": "vae"},
        "gated_full.safetensors": {"url": "https://h/f",
                                   "subdir": "checkpoints",
                                   "flag": "download_ltx2_19b",
                                   "variant": "full"},
        "gated_fp8.safetensors": {"url": "https://h/8",
                                  "subdir": "checkpoints",
                                  "flag": "download_ltx2_19b",
                                  "variant": "fp8"},
        "gated_te.safetensors": {"url": "https://h/t",
                                 "subdir": "text_encoders",
                                 "flag": "download_ltx2_19b"},
    }
    off = select(greg, {"download_ltx2_19b": "false"}, lightweight_fp8=False)
    assert set(off) == {"always.safetensors"}, off
    on_full = select(greg, {"download_ltx2_19b": "true"},
                     lightweight_fp8=False)
    assert set(on_full) == {"always.safetensors", "gated_full.safetensors",
                            "gated_te.safetensors"}, on_full
    on_fp8 = select(greg, {"download_ltx2_19b": "true"}, lightweight_fp8=True)
    assert set(on_fp8) == {"always.safetensors", "gated_fp8.safetensors",
                           "gated_te.safetensors"}, on_fp8
    # widened opt-in truthiness (CONTRACTS.md section 3)
    on_yes = select(greg, {"download_ltx2_19b": "yes"}, lightweight_fp8=False)
    assert set(on_yes) == set(on_full), on_yes

    # disable_flag: opt-out, shipped by default, dropped when exactly "true"
    dreg = {
        "keep.safetensors": {"url": "https://h/k", "subdir": "loras"},
        "ic.safetensors": {"url": "https://h/i", "subdir": "loras",
                           "disable_flag": "disable_ic_loras"},
    }
    assert set(select(dreg, {}, False)) == {"keep.safetensors",
                                            "ic.safetensors"}
    assert set(select(dreg, {"disable_ic_loras": "true"}, False)) == {
        "keep.safetensors"}

    # gated: queued only when HF_TOKEN is present
    greg2 = {
        "open.safetensors": {"url": "https://h/o", "subdir": "loras"},
        "gated.safetensors": {"url": "https://h/g", "subdir": "loras",
                              "gated": True},
    }
    assert set(select(greg2, {}, False)) == {"open.safetensors"}
    assert set(select(greg2, {"HF_TOKEN": "hf_x"}, False)) == {
        "open.safetensors", "gated.safetensors"}
    assert set(select(greg2, {"HF_TOKEN": "  "}, False)) == {
        "open.safetensors"}  # blank token ignored

    # flag + gated compose with AND; either one missing drops the entry
    creg = {
        "ltx25.safetensors": {"url": "https://h/2.5",
                              "subdir": "diffusion_models",
                              "flag": "download_ltx25", "gated": True},
        "ltx25_open.safetensors": {"url": "https://h/e2b",
                                   "subdir": "text_encoders",
                                   "flag": "download_ltx25"},
    }
    assert set(select(creg, {}, False)) == set()
    assert set(select(creg, {"HF_TOKEN": "hf_x"}, False)) == set()
    assert set(select(creg, {"download_ltx25": "true"}, False)) == {
        "ltx25_open.safetensors"}
    assert set(select(creg, {"download_ltx25": "true", "HF_TOKEN": "hf_x"},
                      False)) == {"ltx25.safetensors",
                                  "ltx25_open.safetensors"}

    # on_by_default: ships unless the env is a literal "false"
    breg = {
        "always.safetensors": {"url": "https://h/a", "subdir": "vae"},
        "ltx23.safetensors": {"url": "https://h/23", "subdir": "checkpoints",
                              "flag": "download_ltx23",
                              "on_by_default": True},
        "optin.safetensors": {"url": "https://h/o", "subdir": "checkpoints",
                              "flag": "download_ltx25"},
    }
    assert set(select(breg, {}, False)) == {"always.safetensors",
                                            "ltx23.safetensors"}
    assert set(select(breg, {"download_ltx23": "true"}, False)) == {
        "always.safetensors", "ltx23.safetensors"}
    assert set(select(breg, {"download_ltx23": "false"}, False)) == {
        "always.safetensors"}
    assert set(select(breg, {"download_ltx23": "FALSE"}, False)) == {
        "always.safetensors"}
    # a nonsense value must not silently drop the default set
    assert set(select(breg, {"download_ltx23": "no"}, False)) == {
        "always.safetensors", "ltx23.safetensors"}
    # opt-in flags are unaffected by the new semantic
    assert set(select(breg, {"download_ltx25": "true"}, False)) == {
        "always.safetensors", "ltx23.safetensors", "optin.safetensors"}

    # min_size_mb: a small file above its own floor is skippable, and the
    # floor rides as the manifest's third field
    sreg = {"tiny.safetensors": {"url": "https://h/t",
                                 "subdir": "model_patches",
                                 "min_size_mb": 1}}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        tiny = root / "model_patches/tiny.safetensors"
        tiny.parent.mkdir(parents=True)
        tiny.write_bytes(b"\0" * (2 * 1024 * 1024))  # above its 1 MB floor
        lines, skipped = build(sreg, root)
        assert skipped == ["tiny.safetensors"] and lines == [], (skipped,
                                                                 lines)
        tiny.write_bytes(b"\0" * 1024)  # 1 KB: truncated, must re-queue
        lines, skipped = build(sreg, root)
        assert skipped == [] and len(lines) == 1, (skipped, lines)
        assert lines[0].split("\t")[2] == "1", lines

    # swap state: managed covers every profile, swaps target the selection,
    # inactive groups contribute managed names but no swaps or queue
    tpl = {"swap_groups": [
        {"env": "q", "default": "int8", "flags": ["download_x"],
         "profiles": {"int8": {"dit": "a_int8.safetensors"},
                      "fp8": {"dit": "a_fp8.safetensors"}}},
        {"env": "p", "default": "bf16", "flags": ["download_y"],
         "profiles": {"bf16": {"m": "b_bf16.safetensors"},
                      "fp8": {"m": "b_fp8.safetensors"}}},
    ]}
    sreg2 = {n: {"url": f"https://h/{n}", "subdir": "x"}
             for n in ("a_int8.safetensors", "a_fp8.safetensors",
                       "b_bf16.safetensors", "b_fp8.safetensors")}
    swaps, managed, sel = build_swap_state(tpl, {"q": "fp8"},
                                           {"download_x"}, sreg2)
    assert swaps == {"a_int8.safetensors": "a_fp8.safetensors"}, swaps
    assert managed == set(sreg2), managed
    assert sel == ["a_fp8.safetensors"], sel
    # selected profile file missing from the registry is a config error
    try:
        build_swap_state(tpl, {"q": "fp8"}, {"download_x"}, {})
        raise AssertionError("missing profile file did not raise")
    except ConfigError:
        pass

    # rewrite: loader widgets AND properties.models, top level AND subgraph
    # definitions AND the instance node's promoted widgets
    doc = {
        "nodes": [
            {"id": 1, "widgets_values": ["a_int8.safetensors", 3],
             "properties": {"models": [
                 {"name": "a_int8.safetensors", "url": "old"}]}},
            {"id": 2, "type": "sg", "widgets_values": ["a_int8.safetensors"]},
        ],
        "definitions": {"subgraphs": [{"id": "sg", "nodes": [
            {"id": 10, "widgets_values": ["a_int8.safetensors"],
             "properties": {"models": [
                 {"name": "a_int8.safetensors", "url": "old"}]}},
        ]}]},
    }
    rewrite_doc(doc, {"a_int8.safetensors": "a_fp8.safetensors"}, sreg2)
    assert doc["nodes"][0]["widgets_values"] == ["a_fp8.safetensors", 3]
    assert doc["nodes"][0]["properties"]["models"][0] == {
        "name": "a_fp8.safetensors",
        "url": "https://h/a_fp8.safetensors"}
    assert doc["nodes"][1]["widgets_values"] == ["a_fp8.safetensors"]
    inner = doc["definitions"]["subgraphs"][0]["nodes"][0]
    assert inner["widgets_values"] == ["a_fp8.safetensors"]
    assert inner["properties"]["models"][0]["url"] == (
        "https://h/a_fp8.safetensors")

    # flag truthiness (CONTRACTS.md section 3)
    for v in ("1", "true", "yes", "on", " TRUE "):
        assert flag_enabled({"f": v}, "f", False), v
    for v in ("false", "0", "", "no"):
        assert not flag_enabled({"f": v}, "f", False), v
    assert not flag_enabled({}, "f", False)
    assert flag_enabled({}, "f", True)
    assert flag_enabled({"f": "typo"}, "f", True)
    assert not flag_enabled({"f": " False "}, "f", True)
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        sys.exit(main())
