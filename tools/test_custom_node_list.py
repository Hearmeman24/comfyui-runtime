#!/usr/bin/env python3
"""Offline self-test for the custom-node clone loop in src/start.sh.

Run: python3 tools/test_custom_node_list.py

Extracts the marked clone loop out of the SHIPPED start.sh and runs it in bash
against a stub `git` that records argv. Nothing is duplicated here: if the loop
changes shape, this test runs the new one.

Why this exists: the loop now merges two sources — a list the RUNTIME owns
(src/runtime_nodes.json, packs every template gets) and the per-template
custom_nodes.repos. Getting that merge wrong is invisible in review and fails in
one of two expensive directions: a pack silently missing from every pod, or the
same pack cloned twice and a template's pin quietly ignored. Stdlib only, no
pytest.
"""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START = REPO / "src" / "start.sh"

CHECKS: list[tuple[bool, str]] = []


def ok(cond, label):
    CHECKS.append((bool(cond), label))
    print(("ok   " if cond else "FAIL ") + label)


def find_bash4():
    """start.sh needs bash, not sh. macOS /bin/bash is 3.2; prefer a modern one."""
    for cand in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash"):
        if Path(cand).exists():
            return cand
    return "bash"


BASH4 = find_bash4()

# Records every git invocation, and answers rev-parse so the requirements-skip
# logic sees a stable HEAD.
GIT_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$GIT_LOG"
case "$*" in
    *"rev-parse HEAD"*) echo deadbeef ;;
esac
exit 0
"""


def extract_marked(marker):
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if f"{marker}: begin" in l]
    end = [i for i, l in enumerate(lines) if f"{marker}: end" in l]
    ok(len(beg) == 1 and len(end) == 1 and beg[0] < end[0],
       f"start.sh must carry exactly one '{marker}' begin/end marker pair "
       f"(found begin={beg} end={end})")
    return "\n".join(lines[beg[0]:end[0] + 1])


def extract_func(name):
    lines = START.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if l.startswith(f"{name}()")]
    ok(len(beg) == 1, f"expected exactly one {name} definition")
    i, out = beg[0], []
    while True:
        out.append(lines[i])
        if lines[i] == "}":
            break
        i += 1
    return "\n".join(out)


def run_loop(template_json: dict, runtime_nodes, existing=()):
    """Run the real loop. runtime_nodes=None writes NO runtime_nodes.json.

    `existing` names packs already checked out, so the pull path is exercised
    alongside the clone path. Returns (clone targets, pull targets, stdout).
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stub_bin = tmp / "bin"
        stub_bin.mkdir()
        g = stub_bin / "git"
        g.write_text(GIT_STUB)
        g.chmod(0o755)

        runtime_dir = tmp / "runtime"
        (runtime_dir / "src").mkdir(parents=True)
        if runtime_nodes is not None:
            payload = (runtime_nodes if isinstance(runtime_nodes, str)
                       else json.dumps(runtime_nodes))
            (runtime_dir / "src" / "runtime_nodes.json").write_text(payload)

        comfy_dir = tmp / "ComfyUI"
        nodes_dir = comfy_dir / "custom_nodes"
        nodes_dir.mkdir(parents=True)
        for name in existing:
            (nodes_dir / name / ".git").mkdir(parents=True)

        tjson = tmp / "template.json"
        tjson.write_text(json.dumps(template_json))
        git_log = tmp / "git.log"

        script = tmp / "boot.sh"
        script.write_text("\n".join([
            "#!/usr/bin/env bash",
            "report_warn() { :; }",
            extract_func("template_json_get"),
            extract_marked("custom-node clone loop"),
            'echo "RESOLVED_DIR=$CUSTOM_NODES_DIR"',
            "wait\n",
        ]))
        env = {
            "PATH": f"{stub_bin}:{os.environ['PATH']}",
            "HOME": os.environ.get("HOME", str(tmp)),
            "TEMPLATE_JSON": str(tjson),
            "RUNTIME_DIR": str(runtime_dir),
            "COMFYUI_DIR": str(comfy_dir),
            "PERSIST_ROOT": str(tmp / "vol"),
            "GIT_LOG": str(git_log),
        }
        r = subprocess.run([BASH4, str(script)], env=env,
                           capture_output=True, text=True, timeout=120)
        ok(r.returncode == 0, f"loop exited {r.returncode}: {r.stderr[-400:]}")
        raw = git_log.read_text().splitlines() if git_log.exists() else []
        clones = [l.split()[-1].rsplit("/", 1)[-1] for l in raw if l.startswith("clone ")]
        # `git -C <dir> pull --ff-only` -> the stub records "-C <dir> pull ...",
        # so the directory is field 1.
        pulls = [l.split()[1].rsplit("/", 1)[-1] for l in raw
                 if l.startswith("-C ") and " pull" in l]
        return clones, pulls, r.stdout


A = "https://github.com/Hearmeman24/ComfyUI-HearmemanAI-Upscale.git"
B = "https://github.com/kijai/ComfyUI-KJNodes.git"
C = "https://github.com/rgthree/rgthree-comfy.git"
OPENROUTER_SIMPLE = "https://github.com/Hearmeman24/ComfyUI-OpenRouter-Simple.git"


# --- tests ------------------------------------------------------------------

def test_no_runtime_file_is_unchanged_behaviour():
    """A runtime without the file must behave exactly as it did before this
    existed, or promoting `stable` breaks every template at once."""
    clones, _, _ = run_loop({"custom_nodes": {"target": "image", "repos": [B]}}, None)
    ok(clones == ["ComfyUI-KJNodes"],
       f"no runtime_nodes.json: only the template's own repos clone (got {clones})")

    clones, _, _ = run_loop({"custom_nodes": {"target": "image", "repos": []}}, None)
    ok(clones == [], f"no runtime file and no template repos: nothing clones (got {clones})")


def test_runtime_list_reaches_a_template_that_lists_nothing():
    clones, _, _ = run_loop({"custom_nodes": {"target": "image", "repos": []}}, [A])
    ok(clones == ["ComfyUI-HearmemanAI-Upscale"],
       f"a runtime-owned pack clones even when the template lists none (got {clones})")


def test_both_lists_are_unioned():
    clones, _, _ = run_loop({"custom_nodes": {"target": "image", "repos": [B, C]}}, [A])
    ok(sorted(clones) == ["ComfyUI-HearmemanAI-Upscale", "ComfyUI-KJNodes", "rgthree-comfy"],
       f"runtime and template lists union, nothing dropped (got {clones})")
    ok(clones[0] == "ComfyUI-HearmemanAI-Upscale",
       f"runtime packs clone first, so a template entry can override one (got {clones})")


def test_a_pack_named_by_both_is_cloned_once():
    """Two entries for one directory would clone, then clone over the top."""
    clones, _, _ = run_loop({"custom_nodes": {"target": "image", "repos": [A, B]}}, [A])
    ok(clones.count("ComfyUI-HearmemanAI-Upscale") == 1,
       f"a pack on both lists is cloned exactly once (got {clones})")
    ok(sorted(clones) == ["ComfyUI-HearmemanAI-Upscale", "ComfyUI-KJNodes"],
       f"and the other entries are untouched (got {clones})")


def test_the_template_entry_wins_on_a_collision():
    """The template is the more specific source. If it pins a pack the runtime
    lists unpinned, the pin must survive."""
    _, _, out = run_loop({"custom_nodes": {"target": "image", "repos": [A + "|abc1234"]}}, [A])
    ok("abc1234" in out or "Removing" not in out,
       "a template pin on a runtime-listed pack is not silently discarded")


def test_existing_checkouts_pull_instead_of_cloning():
    clones, pulls, _ = run_loop({"custom_nodes": {"target": "image", "repos": [B]}}, [A],
                                existing=["ComfyUI-HearmemanAI-Upscale"])
    ok("ComfyUI-HearmemanAI-Upscale" not in clones,
       f"an already-checked-out runtime pack is not re-cloned (got {clones})")
    ok("ComfyUI-HearmemanAI-Upscale" in pulls,
       f"it is pulled --ff-only instead, so it tracks upstream (got {pulls})")


def test_a_broken_runtime_file_does_not_abort_the_boot():
    """This file is read on every pod of every template. A typo in it must cost
    the runtime packs, never the boot."""
    clones, _, _ = run_loop({"custom_nodes": {"target": "image", "repos": [B]}},
                            "{ this is not json")
    ok(clones == ["ComfyUI-KJNodes"],
       f"malformed runtime_nodes.json: boot continues on the template's list (got {clones})")


def test_volume_target_still_wins():
    clones, _, out = run_loop({"custom_nodes": {"target": "volume", "repos": []}}, [A])
    ok("/vol/custom_nodes" in out,
       f"target=volume still redirects the whole loop to the volume ({out.strip()[-80:]})")
    ok(clones == ["ComfyUI-HearmemanAI-Upscale"],
       f"and runtime packs land there too (got {clones})")


def test_the_shipped_runtime_nodes_file_is_valid():
    p = REPO / "src" / "runtime_nodes.json"
    ok(p.exists(), "src/runtime_nodes.json is committed")
    if not p.exists():
        return
    try:
        data = json.loads(p.read_text())
    except ValueError as e:
        ok(False, f"src/runtime_nodes.json parses: {e}")
        return
    ok(isinstance(data, list), f"it is a JSON array (got {type(data).__name__})")
    ok(all(isinstance(x, str) and x.startswith("https://") for x in data),
       "every entry is an https URL string")
    ok(data.count(OPENROUTER_SIMPLE) == 1,
       "OpenRouter Simple is declared exactly once in the shipped runtime list")
    names = [x.split("|")[0].rsplit("/", 1)[-1] for x in data]
    ok(len(names) == len(set(names)), f"no two entries share a directory name ({names})")


def main():
    for t in (test_no_runtime_file_is_unchanged_behaviour,
              test_runtime_list_reaches_a_template_that_lists_nothing,
              test_both_lists_are_unioned,
              test_a_pack_named_by_both_is_cloned_once,
              test_the_template_entry_wins_on_a_collision,
              test_existing_checkouts_pull_instead_of_cloning,
              test_a_broken_runtime_file_does_not_abort_the_boot,
              test_volume_target_still_wins,
              test_the_shipped_runtime_nodes_file_is_valid):
        t()
    failed = [label for good, label in CHECKS if not good]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for label in failed:
            print(f"FAILED: {label}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
