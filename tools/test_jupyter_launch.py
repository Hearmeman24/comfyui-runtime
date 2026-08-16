#!/usr/bin/env python3
"""Self-check for the JupyterLab launch in src/start.sh.

JupyterLab auth is OPT IN. Set JUPYTER_TOKEN on the pod and JupyterLab asks
for it; leave it unset and JupyterLab stays open to anyone holding the pod
URL, which is how every pod in this family has always run.

Whether JupyterLab runs AT ALL is a per-template choice: `"jupyter": false` in
template.json skips the launch (a private client pod publishes 8188 only, and
not publishing 8888 hides the proxy route but leaves the process running and
bound). The key is opt OUT — absent means launch — so the four public
templates, none of which carry it, are untouched.

Five things get pinned here that nobody can check by eye:

  - with JUPYTER_TOKEN unset the argv is byte for byte the command line the
    family shipped before this change, so the default boot is untouched;
  - with JUPYTER_TOKEN set the token appears NOWHERE in argv (it would show up
    in `ps` for every process on the pod) and NOWHERE in stdout (the whole boot
    log is teed to $NETWORK_VOLUME/comfyui.log, which support asks customers to
    paste into Discord). Jupyter reads the variable out of its own environment:
    jupyter_server/auth/identity.py, IdentityProvider._token_default, checks
    os.getenv("JUPYTER_TOKEN") before it generates anything;
  - BOTH call sites are covered. start.sh launches JupyterLab twice, once per
    branch of the NETWORK_VOLUME check, and covering only one is the obvious
    way to get this wrong;
  - `false` in ANY case disables the launch, and nothing else does. A missing
    key, an unreadable template.json, a missing template.json and any junk
    value all still start JupyterLab, so a typo can never silently take it
    away from a customer (same direction as provisioner.flag_enabled's opt-out
    mode, :55-62). The case-insensitive match is where this switch departs
    from those flags: everywhere else "safe" means keeping the feature, but
    here the feature IS the exposure, so `"jupyter": "False"` quietly starting
    an unauthenticated shell on a client's pod is the bad outcome;
  - the four live templates' REAL template.json files are driven through the
    block and must all still launch.

The REAL source text runs, not a copy of it: the block between the
JUPYTER-LAUNCH markers is cut out of src/start.sh — together with the
template_json_get helper it calls — run by bash with a stub `jupyter-lab`
first on PATH, and the stub records its argv and its environment.

No network, no pod, no real JupyterLab. Run: python3 tools/test_jupyter_launch.py
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START_SH = REPO / "src" / "start.sh"
FAMILY = REPO.parent

MARK_START = "# >>> JUPYTER-LAUNCH"
MARK_END = "# <<< JUPYTER-LAUNCH"

# The live public templates. None of them carries a "jupyter" key, and all four
# must keep launching JupyterLab byte for byte as they do today. They are
# SIBLING repos: present on a dev machine, absent in CI, which checks out
# comfyui-runtime alone. A missing sibling CHECKOUT skips that template (see
# test_live_templates_are_untouched); a checkout that is present but missing
# its template.json is a real failure and still goes red.
LIVE_TEMPLATES = ("comfyui-wan", "comfyui-minimax", "comfyui-qwen-image",
                  "comfyui-ltx2")

SKIPPED = []

NO_FILE = object()  # template.json absent entirely (start.sh:36-40 degraded boot)

# The command line as it stood before opt-in auth landed (src/start.sh:146,149
# at 28a4ca1). The default branch must still produce exactly this argv.
HISTORICAL = (
    "--ip=0.0.0.0 --allow-root --no-browser "
    "--NotebookApp.token='' --NotebookApp.password='' "
    "--ServerApp.allow_origin='*' --ServerApp.allow_credentials=True "
    "--notebook-dir={dir}"
)

SECRET = "s3cr3t-token-value"

STUB = """#!/usr/bin/env bash
: > "$ARGV_FILE"
for a in "$@"; do printf '%s\\n' "$a" >> "$ARGV_FILE"; done
printf 'JUPYTER_TOKEN_IN_ENV=%s\\n' "${JUPYTER_TOKEN-<unset>}" >> "$ENV_FILE"
"""

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    assert cond, msg
    CHECKS += 1


def extract_block() -> str:
    """The launch block, verbatim, out of the shipped start.sh."""
    text = START_SH.read_text()
    start = text.find(MARK_START)
    end = text.find(MARK_END)
    ok(start != -1, f"{MARK_START} marker missing from src/start.sh")
    ok(end > start, f"{MARK_END} marker missing or misplaced in src/start.sh")
    return text[start:end]


def extract_helper() -> str:
    """template_json_get (src/start.sh:44), verbatim.

    The launch block reads the `jupyter` key through it, so the block cannot
    run on its own. Cut the real function out rather than reimplementing it:
    a copy here would diverge and pass while the pod fails.
    """
    text = START_SH.read_text()
    start = text.find("template_json_get() {")
    ok(start != -1, "template_json_get() is gone from src/start.sh")
    end = text.find("\n}\n", start)
    ok(end > start, "template_json_get() has no closing brace at column 0")
    body = text[start:end + 3]
    ok("json.load" in body, f"extracted the wrong text: {body!r}")
    return body


def launch(network_volume: str, token=None, template=None,
           expect_launch=True):
    """Run the real block with a stub jupyter-lab.

    `template` is the template.json contents: a dict (or raw string) to write,
    or NO_FILE for no file at all. None means `{}` — a template.json with no
    `jupyter` key, which is what all four live templates ship.

    Returns (argv, stdout, env); argv is None when nothing was launched.
    """
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        bindir = d / "bin"
        bindir.mkdir()
        stub = bindir / "jupyter-lab"
        stub.write_text(STUB)
        stub.chmod(0o755)

        template_json = d / "template.json"
        if template is NO_FILE:
            pass
        elif isinstance(template, str):
            template_json.write_text(template)
        else:
            template_json.write_text(json.dumps(template or {}))

        argv_file = d / "argv.txt"
        env_file = d / "env.txt"
        script = d / "block.sh"
        # `wait` because start.sh backgrounds JupyterLab; without it the shell
        # can exit before the stub has written anything.
        script.write_text(extract_helper() + "\n" + extract_block() + "\nwait\n")

        env = {
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "ARGV_FILE": str(argv_file),
            "ENV_FILE": str(env_file),
            "NETWORK_VOLUME": network_volume,
            "TEMPLATE_JSON": str(template_json),
            "HOME": str(d),
        }
        if token is not None:
            env["JUPYTER_TOKEN"] = token

        r = subprocess.run(["bash", str(script)], capture_output=True,
                           text=True, env=env)
        ok(r.returncode == 0, r.stdout + r.stderr)
        if expect_launch is None:
            # The caller is asking WHETHER it launched, not asserting it.
            if not argv_file.exists():
                return None, r.stdout, ""
        elif not expect_launch:
            ok(not argv_file.exists(),
               "jupyter-lab was launched but should not have been: "
               + (argv_file.read_text() if argv_file.exists() else ""))
            return None, r.stdout, ""
        ok(argv_file.is_file(), "jupyter-lab was never launched: "
                                + r.stdout + r.stderr)
        argv = argv_file.read_text().splitlines()
        seen_env = env_file.read_text().strip() if env_file.is_file() else ""
        return argv, r.stdout, seen_env


def test_default_boot_is_unchanged():
    """No JUPYTER_TOKEN: identical argv to what the family shipped before."""
    for network_volume, notebook_dir in (("/workspace", "/workspace"),
                                         ("/", "/")):
        argv, out, _ = launch(network_volume)
        expected = shlex.split(HISTORICAL.format(dir=notebook_dir))
        ok(argv == expected,
           f"{network_volume}: argv drifted\n got {argv}\nwant {expected}")
        ok("no login" in out.lower(),
           f"an open JupyterLab must say so in the log: {out}")


def test_token_set_removes_the_no_auth_overrides():
    """JUPYTER_TOKEN set: the two flags that disable auth are gone, and
    nothing else about the command line moves."""
    for network_volume, notebook_dir in (("/workspace", "/workspace"),
                                         ("/", "/")):
        argv, _, _ = launch(network_volume, token=SECRET)
        expected = [a for a in shlex.split(HISTORICAL.format(dir=notebook_dir))
                    if not a.startswith(("--NotebookApp.token",
                                         "--NotebookApp.password"))]
        ok(argv == expected,
           f"{network_volume}: argv drifted\n got {argv}\nwant {expected}")
        ok(not any("token" in a.lower() or "password" in a.lower()
                   for a in argv),
           f"no auth flag may survive on the command line: {argv}")


def test_token_never_reaches_argv_or_the_log():
    """The hard constraint. The boot log is pasted into Discord for support."""
    for network_volume in ("/workspace", "/"):
        argv, out, seen_env = launch(network_volume, token=SECRET)
        ok(not any(SECRET in a for a in argv),
           f"token leaked into argv (visible in ps): {argv}")
        ok(SECRET not in out, f"token leaked into the boot log: {out}")
        # Not even a masked or partial form.
        ok(SECRET[:6] not in out, f"partial token in the boot log: {out}")
        # And it did reach Jupyter, by the only route that is not argv.
        ok(seen_env == f"JUPYTER_TOKEN_IN_ENV={SECRET}",
           f"jupyter-lab did not inherit JUPYTER_TOKEN: {seen_env!r}")


def test_empty_token_is_treated_as_unset():
    """A customer who adds the variable and leaves it blank gets today's
    behaviour, not a JupyterLab nobody can log into."""
    argv, _, _ = launch("/workspace", token="")
    ok(argv == shlex.split(HISTORICAL.format(dir="/workspace")),
       f"empty JUPYTER_TOKEN must behave exactly as unset: {argv}")


def test_both_call_sites_go_through_one_function():
    """The failure mode this change invites: fixing one branch of the
    NETWORK_VOLUME check and leaving the other wide open."""
    text = START_SH.read_text()
    invocations = [ln for ln in text.splitlines()
                   if "jupyter-lab" in ln and not ln.lstrip().startswith("#")]
    ok(len(invocations) == 1,
       f"jupyter-lab must be invoked in exactly one place: {invocations}")
    call_sites = [ln for ln in text.splitlines()
                  if ln.strip().startswith("start_jupyter ")]
    ok(len(call_sites) == 2,
       f"both NETWORK_VOLUME branches must launch JupyterLab: {call_sites}")
    ok("--notebook-dir=/workspace" not in text
       and any(ln.strip() == "start_jupyter /workspace" for ln in call_sites),
       "the volume branch must pass /workspace through the function")
    ok(any(ln.strip() == "start_jupyter /" for ln in call_sites),
       "the root branch must pass / through the function")


def test_absent_jupyter_key_still_launches():
    """The shape all four live templates are in today. An absent key reads as
    the empty string out of template_json_get, and empty must mean launch."""
    for template, label in ((None, "template.json with no jupyter key"),
                            (NO_FILE, "no template.json at all"),
                            ("{ not json", "unreadable template.json")):
        for network_volume, notebook_dir in (("/workspace", "/workspace"),
                                             ("/", "/")):
            argv, out, _ = launch(network_volume, template=template)
            ok(argv == shlex.split(HISTORICAL.format(dir=notebook_dir)),
               f"{label} ({network_volume}): argv drifted: {argv}")
            ok("Starting JupyterLab" in out,
               f"{label}: the launch must still be announced: {out}")


def test_jupyter_false_does_not_launch():
    """A private client pod. Not publishing 8888 hides the proxy route; only
    this stops the process from running and binding at all."""
    for network_volume in ("/workspace", "/"):
        _, out, _ = launch(network_volume, template={"jupyter": False},
                           expect_launch=False)
        ok("jupyter" in out.lower() and "disabled" in out.lower(),
           f"skipping the launch must say so, once and clearly: {out!r}")
        ok("Starting JupyterLab" not in out,
           f"the log must not claim a launch that did not happen: {out!r}")
        ok(len([ln for ln in out.splitlines() if ln.strip()]) == 1,
           f"one line, not a paragraph (CLAUDE.md section 6): {out!r}")


def test_false_in_any_case_disables():
    """template_json_get prints a JSON boolean as lowercase true/false, so the
    boolean and the string "false" arrive identically. Case and surrounding
    whitespace are both ignored, which is the one place this switch departs
    from the family's opt-out flags: everywhere else "safe" means keeping the
    feature, but here the feature IS the exposure — an unauthenticated shell
    on a client's pod. `"jupyter": "False"` silently launching JupyterLab is
    the bad outcome, not the safe one. Matching flag_enabled's .strip().lower()
    (src/provisioner.py:57) rather than half of it."""
    for spelling in ("false", "False", "FALSE", "FaLsE",
                     "false ", " false", "\tFalse\n", "  FALSE  "):
        for network_volume in ("/workspace", "/"):
            _, out, _ = launch(network_volume, template={"jupyter": spelling},
                               expect_launch=False)
            ok("disabled" in out.lower(),
               f"jupyter={spelling!r} must disable the launch: {out!r}")
            ok("Starting JupyterLab" not in out,
               f"jupyter={spelling!r}: no launch may be claimed: {out!r}")


def test_jupyter_true_launches():
    """The explicit opt in, byte identical to the default."""
    for network_volume, notebook_dir in (("/workspace", "/workspace"),
                                         ("/", "/")):
        argv, out, _ = launch(network_volume, template={"jupyter": True})
        ok(argv == shlex.split(HISTORICAL.format(dir=notebook_dir)),
           f'"jupyter": true ({network_volume}): argv drifted: {argv}')
        ok("disabled" not in out.lower(), f"true must not disable: {out!r}")


def test_only_false_disables():
    """Opt-out truthiness, the direction the family already uses for base-set
    flags (src/provisioner.py:55-62): nothing but false disables, so a typo
    leaves the customer's JupyterLab where it was. "no"/"0"/"off" are NOT
    synonyms for false here."""
    for junk in ("no", "0", 0, "off", "", None, [], {}, "true", "True"):
        argv, out, _ = launch("/workspace", template={"jupyter": junk})
        ok(argv == shlex.split(HISTORICAL.format(dir="/workspace")),
           f"jupyter={junk!r} is not a literal false and must launch: {argv}")
        ok("disabled" not in out.lower(),
           f"jupyter={junk!r} must not disable: {out!r}")


def test_live_templates_are_untouched():
    """The no-regression claim, made against the real files rather than a
    reconstruction of them: none of the four carries the key, and each one
    still produces the historical command line.

    Reads sibling repos, so it only runs where they are checked out. A sibling
    whose repo directory does not exist at all is skipped and announced — that
    is CI, which clones comfyui-runtime alone. Anything else keeps failing:
    a checkout that exists but has no template.json, and every assertion
    below."""
    for name in LIVE_TEMPLATES:
        if not (FAMILY / name).is_dir():
            SKIPPED.append(f"test_live_templates_are_untouched[{name}] "
                           f"(sibling repo not checked out at {FAMILY / name})")
            continue
        path = FAMILY / name / "template.json"
        ok(path.is_file(), f"{path} is missing; adjust the test, not the pod")
        doc = json.loads(path.read_text())
        ok("jupyter" not in doc,
           f"{name} now carries a jupyter key; this test guards the case where "
           f"it does not")
        for network_volume, notebook_dir in (("/workspace", "/workspace"),
                                             ("/", "/")):
            argv, out, _ = launch(network_volume, template=doc)
            ok(argv == shlex.split(HISTORICAL.format(dir=notebook_dir)),
               f"{name} ({network_volume}): argv drifted: {argv}")
            ok("Starting JupyterLab" in out,
               f"{name}: JupyterLab must still launch: {out!r}")


def test_the_note_agrees_with_the_launch():
    """Two implementations of one switch: the shell gate in start.sh decides
    whether JupyterLab runs, and boot_report.jupyter_enabled decides whether
    the welcome note tells the customer to open port 8888. They must never
    disagree — a note advertising a port nothing answers on is the bug this
    pins (found by the skeleton slice against src/note_welcome.md).
    """
    sys.path.insert(0, str(REPO / "src"))
    import boot_report

    for value in (False, True, "false", "False", "FALSE", "FaLsE", "true",
                  "no", "off", "0", 0, "", None, [], {},
                  "false ", " false", "\tFalse\n", "fa lse", " ", "f alse "):
        argv, _, _ = launch("/workspace", template={"jupyter": value},
                            expect_launch=None)
        shell_launched = argv is not None
        python_says = boot_report.jupyter_enabled({"jupyter": value})
        ok(shell_launched == python_says,
           f"jupyter={value!r}: start.sh launched={shell_launched} but "
           f"boot_report.jupyter_enabled={python_says}; the note and the pod "
           f"would disagree")
    # And the absent key, which is what all four live templates ship.
    ok(boot_report.jupyter_enabled({}) is True,
       "an absent jupyter key must read as enabled")
    ok(boot_report.jupyter_enabled(None) is True,
       "no template.json at all must read as enabled")


def test_the_key_is_allowlisted_in_the_validator():
    """Ordering, non-negotiable: tools/validate_models.py hard-errors on an
    unknown top-level key, so a template.json may only carry `jupyter` once
    the DEPLOYED validator knows it (validate_models.py:409-412)."""
    sys.path.insert(0, str(REPO / "tools"))
    import validate_models as vm
    ok("jupyter" in vm.TEMPLATE_KEYS,
       "'jupyter' must be in TEMPLATE_KEYS or every template using it goes red")


def main():
    test_default_boot_is_unchanged()
    test_token_set_removes_the_no_auth_overrides()
    test_token_never_reaches_argv_or_the_log()
    test_empty_token_is_treated_as_unset()
    test_both_call_sites_go_through_one_function()
    test_absent_jupyter_key_still_launches()
    test_jupyter_false_does_not_launch()
    test_false_in_any_case_disables()
    test_jupyter_true_launches()
    test_only_false_disables()
    test_live_templates_are_untouched()
    test_the_note_agrees_with_the_launch()
    test_the_key_is_allowlisted_in_the_validator()
    for note in SKIPPED:
        print(f"jupyter launch self-test: SKIPPED {note}")
    if len(SKIPPED) == len(LIVE_TEMPLATES):
        print("jupyter launch self-test: WARNING none of the four live "
              "templates was checked — no sibling repo is checked out beside "
              "comfyui-runtime, so the no-regression guard did not run here")
    print(f"jupyter launch self-test: all good ({CHECKS} assertions, "
          f"{len(SKIPPED)} skipped)")


if __name__ == "__main__":
    main()
