#!/usr/bin/env python3
"""Self-check for the JupyterLab launch in src/start.sh.

JupyterLab auth is OPT IN. Set JUPYTER_TOKEN on the pod and JupyterLab asks
for it; leave it unset and JupyterLab stays open to anyone holding the pod
URL, which is how every pod in this family has always run.

Three things get pinned here that nobody can check by eye:

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
    way to get this wrong.

The REAL source text runs, not a copy of it: the block between the
JUPYTER-LAUNCH markers is cut out of src/start.sh, run by bash with a stub
`jupyter-lab` first on PATH, and the stub records its argv and its environment.

No network, no pod, no real JupyterLab. Run: python3 tools/test_jupyter_launch.py
"""
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START_SH = REPO / "src" / "start.sh"

MARK_START = "# >>> JUPYTER-LAUNCH"
MARK_END = "# <<< JUPYTER-LAUNCH"

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


def launch(network_volume: str, token=None):
    """Run the real block with a stub jupyter-lab. Returns (argv, stdout, env)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        bindir = d / "bin"
        bindir.mkdir()
        stub = bindir / "jupyter-lab"
        stub.write_text(STUB)
        stub.chmod(0o755)

        argv_file = d / "argv.txt"
        env_file = d / "env.txt"
        script = d / "block.sh"
        # `wait` because start.sh backgrounds JupyterLab; without it the shell
        # can exit before the stub has written anything.
        script.write_text(extract_block() + "\nwait\n")

        env = {
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "ARGV_FILE": str(argv_file),
            "ENV_FILE": str(env_file),
            "NETWORK_VOLUME": network_volume,
            "HOME": str(d),
        }
        if token is not None:
            env["JUPYTER_TOKEN"] = token

        r = subprocess.run(["bash", str(script)], capture_output=True,
                           text=True, env=env)
        ok(r.returncode == 0, r.stdout + r.stderr)
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


def main():
    test_default_boot_is_unchanged()
    test_token_set_removes_the_no_auth_overrides()
    test_token_never_reaches_argv_or_the_log()
    test_empty_token_is_treated_as_unset()
    test_both_call_sites_go_through_one_function()
    print(f"jupyter launch self-test: all good ({CHECKS} assertions)")


if __name__ == "__main__":
    main()
