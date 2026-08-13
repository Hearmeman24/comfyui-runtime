#!/usr/bin/env python3
"""Self-check for the universal model-path derivation (src/model_paths.py and
the derived-model-paths block in src/start.sh).

The category list for extra_model_paths.yaml is no longer hand-managed: it is
derived at boot from the pinned ComfyUI tree's own folder_paths registry, so
it can never drift from the ComfyUI the image actually runs (ltx2 migration
spec D5). What must hold:

  - every folder_names_and_paths dir under models_dir is emitted as
    "key<TAB>relpath", so the legacy alternate dirs ride under their REAL key:
    clip under text_encoders, unet under diffusion_models, t2i_adapter under
    controlnet (t2i_adapter is NOT in map_legacy and must never be a key)
  - custom_nodes and datasets hang off base_path, not models_dir, and are
    never emitted as models/<cat>
  - on ANY derivation failure the frozen v0.32.0 superset (28 dirs) is
    printed instead (exit 3), never an empty or partial list
  - in start.sh, the mkdir loop and the yaml come from the SAME derived list,
    and template.json extra_model_paths entries stay accepted and additive

The start.sh block is exercised for real: extracted between its begin/end
markers and run under bash with a stubbed ComfyUI tree. No pod, no network.

Same technique, second block: the custom-node clone loop's conditional
requirements install. A pack's requirements install only when the checkout
actually changed , fresh clone, a pull that moved HEAD, or a pin reset that
moved HEAD , never on an "Already up to date" boot, and never on a FAILED
pull (regression: ltx2 v10 reran every pack's install each boot, which kept
clobbering onnxruntime-gpu and forced a 250 MB reinstall). Skipped packs must
leave no entry in PIP_INSTALL_PIDS for the wait loop to choke on. Exercised
with local throwaway git repos and a stubbed pip on PATH.
Run: python3 tools/test_model_paths.py
"""
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MP = REPO / "src" / "model_paths.py"
START = REPO / "src" / "start.sh"

# A real ComfyUI v0.32.0 clone, when one is around (dev machines). CI does not
# have it; those tests skip with a notice. Override with COMFYUI_TREE=<dir>.
REAL_TREE = Path(os.environ.get("COMFYUI_TREE", "")) if os.environ.get(
    "COMFYUI_TREE") else Path("/nonexistent/comfyui-tree")

# The full derivation against v0.32.0's folder_paths.py, in registration
# order: 27 keys, 25 of them models_dir-based, yielding these 28 dirs.
# This is also, verbatim, the frozen fallback model_paths.py must ship.
EXPECTED_V0320 = [
    ("checkpoints", "checkpoints"),
    ("configs", "configs"),
    ("loras", "loras"),
    ("vae", "vae"),
    ("text_encoders", "text_encoders"),
    ("text_encoders", "clip"),
    ("diffusion_models", "unet"),
    ("diffusion_models", "diffusion_models"),
    ("clip_vision", "clip_vision"),
    ("style_models", "style_models"),
    ("embeddings", "embeddings"),
    ("diffusers", "diffusers"),
    ("vae_approx", "vae_approx"),
    ("controlnet", "controlnet"),
    ("controlnet", "t2i_adapter"),
    ("gligen", "gligen"),
    ("upscale_models", "upscale_models"),
    ("latent_upscale_models", "latent_upscale_models"),
    ("hypernetworks", "hypernetworks"),
    ("photomaker", "photomaker"),
    ("classifiers", "classifiers"),
    ("model_patches", "model_patches"),
    ("audio_encoders", "audio_encoders"),
    ("background_removal", "background_removal"),
    ("frame_interpolation", "frame_interpolation"),
    ("geometry_estimation", "geometry_estimation"),
    ("optical_flow", "optical_flow"),
    ("detection", "detection"),
]

# wan's old frozen base list (CONTRACTS.md section 5), minus custom_nodes
# which was never a models/ dir. The derived set must remain a superset so
# every existing volume keeps every category it had (invariant I5).
OLD_BASE_LIST = {
    "checkpoints", "clip", "clip_vision", "controlnet", "diffusion_models",
    "embeddings", "loras", "style_models", "text_encoders", "unet",
    "upscale_models", "latent_upscale_models", "detection", "vae",
}

# Fake ComfyUI tree: multi-dir keys with their alternates, plus the two
# base_path dirs mid-stream to prove exclusion is positional-independent.
STUB_FOLDER_PATHS = textwrap.dedent("""\
    import os
    base_path = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(base_path, "models")
    folder_names_and_paths = {
        "checkpoints": ([os.path.join(models_dir, "checkpoints")], {".safetensors"}),
        "text_encoders": ([os.path.join(models_dir, "text_encoders"),
                           os.path.join(models_dir, "clip")], set()),
        "custom_nodes": ([os.path.join(base_path, "custom_nodes")], set()),
        "diffusion_models": ([os.path.join(models_dir, "unet"),
                              os.path.join(models_dir, "diffusion_models")], set()),
        "datasets": ([os.path.join(base_path, "datasets")], set()),
        "controlnet": ([os.path.join(models_dir, "controlnet"),
                        os.path.join(models_dir, "t2i_adapter")], set()),
        "loras": ([os.path.join(models_dir, "loras")], set()),
    }
""")

EXPECTED_STUB = [
    ("checkpoints", "checkpoints"),
    ("text_encoders", "text_encoders"),
    ("text_encoders", "clip"),
    ("diffusion_models", "unet"),
    ("diffusion_models", "diffusion_models"),
    ("controlnet", "controlnet"),
    ("controlnet", "t2i_adapter"),
    ("loras", "loras"),
]

CHECKS = 0
SKIPPED = []


def find_bash4():
    """start.sh needs bash 4+ (declare -A, already relied on elsewhere in the
    script). The pod (Ubuntu 24.04) and CI (cimg/python) both ship bash 5;
    macOS /bin/bash is 3.2, so dev machines probe for a newer one."""
    for cand in ("bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        try:
            r = subprocess.run([cand, "-c", "declare -A __x=()"],
                               capture_output=True)
        except OSError:
            continue
        if r.returncode == 0:
            return cand
    return None


BASH4 = find_bash4()


def ok(cond, msg):
    global CHECKS
    assert cond, msg
    CHECKS += 1


def run_mp(*args):
    r = subprocess.run([sys.executable, str(MP), *args],
                       capture_output=True, text=True)
    pairs = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        k, _, v = line.partition("\t")
        pairs.append((k, v))
    return r.returncode, pairs, r.stderr


def make_stub_tree(root, folder_paths_src=STUB_FOLDER_PATHS):
    tree = Path(root) / "ComfyUI"
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "folder_paths.py").write_text(folder_paths_src)
    return tree


# --- model_paths.py ---------------------------------------------------------

def test_fallback_flag_is_the_frozen_v0320_list():
    rc, pairs, _ = run_mp("--fallback")
    ok(rc == 0, f"--fallback must exit 0, got {rc}")
    ok(pairs == EXPECTED_V0320,
       f"--fallback must print the frozen v0.32.0 list verbatim, got {pairs}")


def test_fallback_invariants():
    rc, pairs, _ = run_mp("--fallback")
    keys = {k for k, _ in pairs}
    dirs = [v for _, v in pairs]
    ok(len(pairs) == 28, f"expected 28 dirs, got {len(pairs)}")
    ok(len(set(pairs)) == len(pairs), "no duplicate (key, dir) pairs")
    for legacy in ("clip", "unet", "t2i_adapter"):
        ok(legacy not in keys, f"legacy alternate '{legacy}' must never be a key")
    ok(("text_encoders", "clip") in pairs, pairs)
    ok(("diffusion_models", "unet") in pairs, pairs)
    ok(("controlnet", "t2i_adapter") in pairs, pairs)
    for base_only in ("custom_nodes", "datasets"):
        ok(base_only not in keys and base_only not in dirs,
           f"base_path dir '{base_only}' must not be emitted as models/<cat>")
    ok(OLD_BASE_LIST <= set(dirs),
       f"derived set must stay a superset of wan's old base list: "
       f"missing {OLD_BASE_LIST - set(dirs)}")


def test_real_tree_derivation():
    if not (REAL_TREE / "folder_paths.py").is_file():
        SKIPPED.append("test_real_tree_derivation (no ComfyUI tree at "
                       f"{REAL_TREE})")
        return
    rc, pairs, err = run_mp(str(REAL_TREE))
    ok(rc == 0, f"derivation against the real tree must exit 0: {rc} {err}")
    ok(pairs == EXPECTED_V0320,
       f"real-tree derivation diverged from the expected v0.32.0 list:\n"
       f"got      {pairs}\nexpected {EXPECTED_V0320}")


def test_stub_tree_derivation():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_stub_tree(tmp)
        rc, pairs, err = run_mp(str(tree))
        ok(rc == 0, f"stub derivation must exit 0: {rc} {err}")
        ok(pairs == EXPECTED_STUB,
           f"got {pairs}\nexpected {EXPECTED_STUB}")


def test_missing_tree_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        rc, pairs, err = run_mp(tmp)  # no folder_paths.py at all
        ok(rc == 3, f"failed derivation must exit 3, got {rc}")
        ok(pairs == EXPECTED_V0320, "fallback list must be printed on failure")
        ok("fallback" in err.lower(), f"stderr must say the fallback ran: {err}")


def test_broken_tree_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_stub_tree(tmp, 'raise RuntimeError("boom at import")\n')
        rc, pairs, _ = run_mp(str(tree))
        ok(rc == 3, f"an import-time crash must exit 3, got {rc}")
        ok(pairs == EXPECTED_V0320, "fallback list must be printed on failure")


def test_bad_usage_still_prints_the_list():
    rc, pairs, _ = run_mp()  # no args: never emit nothing, never abort
    ok(rc == 3, f"bad usage must exit 3, got {rc}")
    ok(pairs == EXPECTED_V0320, "fallback list must be printed on bad usage")


# --- the start.sh block -----------------------------------------------------

def extract_block():
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if "derived model paths: begin" in l]
    end = [i for i, l in enumerate(lines) if "derived model paths: end" in l]
    ok(len(beg) == 1 and len(end) == 1 and beg[0] < end[0],
       "start.sh must carry exactly one derived-model-paths begin/end marker pair")
    return "\n".join(lines[beg[0]:end[0] + 1])


def run_block(comfy_dir, extras="", network_volume=None):
    """Run the real start.sh block with stubbed surroundings. Returns
    (persist_root, comfy_dir, warns, flag)."""
    tmp = tempfile.mkdtemp(prefix="mp_block_")
    nv = network_volume if network_volume is not None else os.path.join(tmp, "nv")
    persist = os.path.join(tmp, "nv", "ComfyUI")
    os.makedirs(persist, exist_ok=True)
    warn_file = os.path.join(tmp, "warns")
    Path(warn_file).touch()
    prelude = (
        'report_warn() { printf "%s\\n" "$1" >> "$WARN_FILE"; }\n'
        'template_json_get() {\n'
        '  if [ "$1" = "extra_model_paths" ] && [ -n "${EXTRAS:-}" ]; then\n'
        '    printf "%s\\n" $EXTRAS\n'
        '  fi\n'
        '}\n'
    )
    footer = '\nprintf "FLAG=%s\\n" "$EXTRA_PATHS_FLAG"\n'
    script = os.path.join(tmp, "block.sh")
    Path(script).write_text(prelude + extract_block() + footer)
    env = {
        "PATH": os.environ["PATH"],
        "RUNTIME_DIR": str(REPO),
        "COMFYUI_DIR": str(comfy_dir),
        "PERSIST_ROOT": persist,
        "NETWORK_VOLUME": nv,
        "EXTRAS": extras,
        "WARN_FILE": warn_file,
    }
    r = subprocess.run([BASH4, script], env=env, capture_output=True, text=True)
    ok(r.returncode == 0, f"block exited {r.returncode}: {r.stderr}")
    warns = Path(warn_file).read_text().splitlines()
    flag = ""
    for line in r.stdout.splitlines():
        if line.startswith("FLAG="):
            flag = line[len("FLAG="):]
    return persist, comfy_dir, warns, flag


def parse_yaml(comfy_dir):
    """Parse the block's generated yaml the way ComfyUI's
    load_extra_path_config consumes it: values split on newlines, every dir
    registered under its yaml key. Returns (base_path, {key: [dirs]})."""
    text = (Path(comfy_dir) / "extra_model_paths.yaml").read_text()
    lines = text.splitlines()
    ok(lines[0] == "network_volume:", lines[:1])
    base_path = None
    mapping = {}
    for line in lines[1:]:
        m = re.match(r'^    base_path: (.*)$', line)
        if m:
            base_path = m.group(1)
            continue
        m = re.match(r'^    ([A-Za-z0-9_]+): "(.*)"$', line)
        if m:
            key, value = m.group(1), m.group(2)
            ok(key not in mapping, f"duplicate yaml key {key}")
            # A double-quoted yaml scalar turns \n into a newline; ComfyUI
            # then splits the loaded value on newlines (utils/extra_config.py).
            mapping[key] = value.split("\\n")
            continue
        m = re.match(r'^    custom_nodes: custom_nodes$', line)
        if m:
            ok("custom_nodes" not in mapping, "duplicate custom_nodes key")
            mapping["custom_nodes"] = ["custom_nodes"]
            continue
        ok(False, f"unrecognized yaml line: {line!r}")
    ok(base_path is not None, "yaml must declare base_path")
    return base_path, mapping


def yaml_model_dirs(mapping):
    dirs = []
    for key, values in mapping.items():
        if key == "custom_nodes":
            continue
        for v in values:
            ok(v.startswith("models/"),
               f"every {key} entry must live under models/: {v}")
            dirs.append(v[len("models/"):])
    return dirs


def created_model_dirs(persist):
    root = Path(persist) / "models"
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def test_block_mkdir_and_yaml_from_same_list():
    if BASH4 is None:
        SKIPPED.append("test_block_mkdir_and_yaml_from_same_list (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_stub_tree(tmp)
        persist, comfy, warns, flag = run_block(
            tree, extras="ipadapter checkpoints clip custom_nodes")
        base_path, mapping = parse_yaml(comfy)
        ok(base_path == persist, (base_path, persist))
        dirs = yaml_model_dirs(mapping)
        # Additive extras: a NEW category lands; already-derived names
        # (checkpoints is a key, clip is text_encoders' alternate dir) and
        # custom_nodes never duplicate.
        expected = sorted([v for _, v in EXPECTED_STUB] + ["ipadapter"])
        ok(sorted(dirs) == expected, f"yaml dirs {sorted(dirs)} != {expected}")
        ok(dirs.count("clip") == 1 and dirs.count("checkpoints") == 1, dirs)
        # The mkdir loop and the yaml are the SAME list: every declared path
        # exists, and nothing extra was created.
        ok(created_model_dirs(persist) == expected,
           f"created {created_model_dirs(persist)} != declared {expected}")
        # Alternates land under their real keys, never as keys.
        ok(mapping["text_encoders"] == ["models/text_encoders", "models/clip"],
           mapping.get("text_encoders"))
        ok(mapping["diffusion_models"] == ["models/unet", "models/diffusion_models"],
           mapping.get("diffusion_models"))
        ok(mapping["controlnet"] == ["models/controlnet", "models/t2i_adapter"],
           mapping.get("controlnet"))
        for legacy in ("clip", "unet", "t2i_adapter"):
            ok(legacy not in mapping, f"'{legacy}' must not be a yaml key")
        # custom_nodes keeps its bespoke base_path line and is never models/.
        ok(mapping["custom_nodes"] == ["custom_nodes"], mapping.get("custom_nodes"))
        ok("custom_nodes" not in dirs and "datasets" not in dirs, dirs)
        ok(flag == f"--extra-model-paths-config {comfy}/extra_model_paths.yaml",
           flag)
        ok(warns == [], f"healthy derivation must not warn: {warns}")


def test_block_falls_back_when_derivation_fails():
    if BASH4 is None:
        SKIPPED.append("test_block_falls_back_when_derivation_fails (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "ComfyUI"
        empty.mkdir()
        persist, comfy, warns, flag = run_block(empty)
        _, mapping = parse_yaml(comfy)
        dirs = yaml_model_dirs(mapping)
        ok(sorted(dirs) == sorted(v for _, v in EXPECTED_V0320),
           f"fallback must declare the full frozen list, got {sorted(dirs)}")
        ok(created_model_dirs(persist) == sorted(set(v for _, v in EXPECTED_V0320)),
           "fallback mkdir list must match the yaml")
        ok(any("fallback" in w.lower() for w in warns),
           f"the boot report must learn the fallback ran: {warns}")
        ok(flag.startswith("--extra-model-paths-config"), flag)


def test_block_no_network_volume():
    if BASH4 is None:
        SKIPPED.append("test_block_no_network_volume (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tree = make_stub_tree(tmp)
        # Pre-plant a stale yaml: the no-volume branch must remove it.
        (tree / "extra_model_paths.yaml").write_text("stale\n")
        persist, comfy, warns, flag = run_block(tree, network_volume="/")
        ok(not (Path(comfy) / "extra_model_paths.yaml").exists(),
           "no network volume: the yaml must be removed")
        ok(flag == "", f"no network volume: flag must stay empty, got {flag!r}")
        # Dirs still exist so nothing downstream trips on a missing path.
        ok(created_model_dirs(persist) == sorted(set(v for _, v in EXPECTED_STUB)),
           created_model_dirs(persist))


# --- the custom-node clone loop block ---------------------------------------

NODES_PRELUDE = (
    'report_warn() { printf "%s\\n" "$1" >> "$WARN_FILE"; }\n'
    'template_json_get() {\n'
    '  case "$1" in\n'
    '    custom_nodes.target) printf "%s" "image" ;;\n'
    '    custom_nodes.repos)\n'
    '      [ -n "${REPO_ENTRIES:-}" ] && printf "%s\\n" $REPO_ENTRIES ;;\n'
    '  esac\n'
    '  return 0\n'
    '}\n'
)

# The real wait loop's pattern (start.sh, pre-launch): a skipped install must
# not leave an entry behind that wait would then choke on.
NODES_FOOTER = (
    '\nfor name in "${!PIP_INSTALL_PIDS[@]}"; do\n'
    '  if wait "${PIP_INSTALL_PIDS[$name]}"; then\n'
    '    printf "WAITED=%s\\n" "$name"\n'
    '  else\n'
    '    printf "WAITFAIL=%s\\n" "$name"\n'
    '  fi\n'
    'done\n'
    'printf "PIDS=%s\\n" "${!PIP_INSTALL_PIDS[*]}"\n'
)


def extract_nodes_block():
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if "custom-node clone loop: begin" in l]
    end = [i for i, l in enumerate(lines) if "custom-node clone loop: end" in l]
    ok(len(beg) == 1 and len(end) == 1 and beg[0] < end[0],
       "start.sh must carry exactly one custom-node-clone-loop marker pair")
    return "\n".join(lines[beg[0]:end[0] + 1])


def git_in(cwd, *args):
    r = subprocess.run(["git", "-c", "user.email=ci@example.invalid",
                        "-c", "user.name=ci", "-c", "init.defaultBranch=main",
                        *args], cwd=cwd, capture_output=True, text=True)
    ok(r.returncode == 0, f"git {args} in {cwd} failed: {r.stderr}")
    return r.stdout.strip()


def make_node_repo(tmp, name):
    """A throwaway upstream repo with a requirements.txt, one commit."""
    src = Path(tmp) / "upstream" / name
    src.mkdir(parents=True)
    (src / "requirements.txt").write_text("example-pkg==1.0\n")
    git_in(src, "init", "-q")
    git_in(src, "add", "-A")
    git_in(src, "commit", "-qm", "v1")
    return src


def bump_node_repo(src, msg):
    (src / "CHANGELOG").write_text(msg + "\n")
    git_in(src, "add", "-A")
    git_in(src, "commit", "-qm", msg)
    return git_in(src, "rev-parse", "HEAD")


def run_nodes_block(tmp, entries):
    """One boot of the real clone-loop block: stubbed pip records its calls,
    local-path repo entries, no network. Returns (stdout, pip calls, PID-array
    key list)."""
    tmp = Path(tmp)
    stub_bin = tmp / "bin"
    stub_bin.mkdir(exist_ok=True)
    pip = stub_bin / "pip"
    pip.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PIP_CALLS"\n')
    pip.chmod(0o755)
    pip_calls = tmp / "pip_calls"
    pip_calls.write_text("")
    warn_file = tmp / "warns"
    warn_file.write_text("")
    comfy = tmp / "ComfyUI"
    comfy.mkdir(exist_ok=True)
    script = tmp / "nodes_block.sh"
    script.write_text(NODES_PRELUDE + extract_nodes_block() + NODES_FOOTER)
    env = {
        "PATH": f"{stub_bin}:{os.environ['PATH']}",
        "HOME": os.environ.get("HOME", str(tmp)),
        "COMFYUI_DIR": str(comfy),
        "PERSIST_ROOT": str(tmp / "nv" / "ComfyUI"),
        "REPO_ENTRIES": " ".join(entries),
        "WARN_FILE": str(warn_file),
        "PIP_CALLS": str(pip_calls),
    }
    r = subprocess.run([BASH4, str(script)], env=env,
                       capture_output=True, text=True)
    ok(r.returncode == 0, f"nodes block exited {r.returncode}: {r.stderr}")
    calls = [l for l in pip_calls.read_text().splitlines() if l]
    pids = []
    for line in r.stdout.splitlines():
        if line.startswith("PIDS="):
            pids = line[len("PIDS="):].split()
    return r.stdout, calls, pids


def node_head(tmp, name):
    return git_in(Path(tmp) / "ComfyUI" / "custom_nodes" / name,
                  "rev-parse", "HEAD")


def test_nodes_fresh_clone_installs():
    if BASH4 is None:
        SKIPPED.append("test_nodes_fresh_clone_installs (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = make_node_repo(tmp, "NodePackA")
        out, calls, pids = run_nodes_block(tmp, [str(src)])
        ok(any("NodePackA/requirements.txt" in c for c in calls),
           f"fresh clone must install requirements: {calls}")
        ok(pids == ["NodePackA"], f"install must be waited on: {pids}")
        ok("WAITED=NodePackA" in out, out)


def test_nodes_unchanged_pull_skips_install():
    if BASH4 is None:
        SKIPPED.append("test_nodes_unchanged_pull_skips_install (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = make_node_repo(tmp, "NodePackA")
        run_nodes_block(tmp, [str(src)])  # boot 1: fresh clone
        out, calls, pids = run_nodes_block(tmp, [str(src)])  # boot 2: no change
        ok(calls == [],
           f"'Already up to date' must not reinstall requirements: {calls}")
        ok(pids == [], f"a skipped install must leave no PID entry: {pids}")
        ok("NodePackA" in out and "skip" in out.lower(),
           f"the skip must be said in the boot log: {out}")


def test_nodes_pull_that_moves_head_installs():
    if BASH4 is None:
        SKIPPED.append("test_nodes_pull_that_moves_head_installs (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = make_node_repo(tmp, "NodePackA")
        run_nodes_block(tmp, [str(src)])
        new_sha = bump_node_repo(src, "v2")
        _, calls, pids = run_nodes_block(tmp, [str(src)])
        ok(any("NodePackA/requirements.txt" in c for c in calls),
           f"a pull that moved HEAD must reinstall requirements: {calls}")
        ok(pids == ["NodePackA"], pids)
        ok(node_head(tmp, "NodePackA") == new_sha, "pull must have landed v2")


def test_nodes_pin_already_current_skips_install():
    if BASH4 is None:
        SKIPPED.append("test_nodes_pin_already_current_skips_install (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = make_node_repo(tmp, "NodePackA")
        run_nodes_block(tmp, [str(src)])
        sha = node_head(tmp, "NodePackA")
        out, calls, pids = run_nodes_block(tmp, [f"{src}|{sha}"])
        ok(calls == [],
           f"a pin already checked out must not reinstall: {calls}")
        ok(pids == [], pids)
        ok("skip" in out.lower(), out)


def test_nodes_pin_reset_that_moves_head_installs():
    if BASH4 is None:
        SKIPPED.append("test_nodes_pin_reset_that_moves_head_installs (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = make_node_repo(tmp, "NodePackA")
        run_nodes_block(tmp, [str(src)])
        new_sha = bump_node_repo(src, "v2")
        _, calls, pids = run_nodes_block(tmp, [f"{src}|{new_sha}"])
        ok(any("NodePackA/requirements.txt" in c for c in calls),
           f"a pin reset that moved HEAD must reinstall requirements: {calls}")
        ok(pids == ["NodePackA"], pids)
        ok(node_head(tmp, "NodePackA") == new_sha, "reset must have landed the pin")


def test_nodes_failed_pull_does_not_install():
    if BASH4 is None:
        SKIPPED.append("test_nodes_failed_pull_does_not_install (no bash 4+)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = make_node_repo(tmp, "NodePackA")
        run_nodes_block(tmp, [str(src)])
        sha = node_head(tmp, "NodePackA")
        # Break the remote: the pull now fails, HEAD stays put.
        git_in(Path(tmp) / "ComfyUI" / "custom_nodes" / "NodePackA",
               "remote", "set-url", "origin", str(Path(tmp) / "gone"))
        out, calls, pids = run_nodes_block(tmp, [str(src)])
        ok("git pull failed" in out, out)
        ok(calls == [],
           f"a FAILED pull must not count as moved and reinstall: {calls}")
        ok(pids == [], pids)
        ok(node_head(tmp, "NodePackA") == sha, "HEAD must be untouched")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    for note in SKIPPED:
        print(f"model paths self-test: SKIPPED {note}")
    print(f"model paths self-test: all good ({CHECKS} assertions, "
          f"{len(SKIPPED)} skipped)")


if __name__ == "__main__":
    sys.exit(main())
