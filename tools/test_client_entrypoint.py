#!/usr/bin/env python3
"""Offline self-test for the PRIVATE-CLIENT template entrypoint.

Run: python3 tools/test_client_entrypoint.py

Subject: comfyui-client-skeleton/src/start_script.sh, the block between
`# --- client sync: begin` and `# --- client sync: end`. That block is the
access gate for every private client engagement: it is the only thing that
reads GITHUB_PAT, the only thing that decides whether a revoked token stops a
client's pod, and the only thing standing between the token and
$NETWORK_VOLUME/comfyui.log — the file support asks customers to paste into
Discord.

The REAL source text runs. The block is cut out of the shipped script and run
by bash against a stub `git` (records argv, emits a chosen stderr) and a stub
`sleep` (records, returns instantly). Nothing here reimplements the shell: a
copy of the classifier in this file would pass while the pod fails closed on a
DNS blip or open on a dead token, which are the two ways this can be wrong.

What gets pinned that nobody can check by eye:

  - no GITHUB_PAT -> exit non-zero, the message names the variable, and NO git
    call is made at all;
  - the sync URL carries `x-access-token:<token>`, and the LAST thing written
    to the remote is a tokenless https://github.com/<slug>.git — on the happy
    path AND on every failure exit path, so a client reading their own pod's
    .git/config never finds the credential;
  - GITHUB_PAT is gone from the environment the runtime, ComfyUI and
    JupyterLab inherit;
  - auth-shaped failures FAIL CLOSED. The subtle ones are 401/403/404: git
    wraps them in `unable to access`, which is also on the network list, so a
    plain network allowlist would boot a revoked token's pod off the stale
    on-disk copy. The auth list has to win;
  - network-shaped failures FAIL OPEN — retry, then boot the on-disk copy and
    say it may be stale — including a 5xx, because a GitHub outage is not a
    credential problem;
  - an unrecognised or EMPTY git error fails closed, and the empty case says
    "suspect the container" instead of printing a blank line;
  - the token appears in NO output on ANY path. Asserted over the captured
    stdout+stderr of every single case below, not just one, with a distinctive
    canary token, and with the stub git deliberately echoing the credentialed
    URL back the way git does.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # comfyui-runtime
FAMILY = REPO.parent                                 # ~/src/comfy
# A SIBLING repo: checked out beside comfyui-runtime on a dev machine, absent
# in CI, which clones comfyui-runtime alone. No skeleton checkout at all -> the
# whole suite skips and says so (see main()). A checkout that IS there but is
# missing src/start_script.sh is a broken skeleton and still fails.
SKELETON = FAMILY / "comfyui-client-skeleton"
SCRIPT = SKELETON / "src" / "start_script.sh"

MARKER = "client sync"
SLUG = "Hearmeman24/comfyui-testclient"
BRANCH = "main"

# Distinctive, >=20 chars of [A-Za-z0-9_-] so it is also a target for the
# script's shape-based sed fallback. A leak of this string is unmistakable.
TOKEN = "ghp_CANARYcanary0000LEAKMEtestonly11"

TOKEN_URL = f"https://x-access-token:{TOKEN}@github.com/{SLUG}.git"
CLEAN_URL = f"https://github.com/{SLUG}.git"

CHECKS: list[tuple[bool, str]] = []


def ok(cond, label):
    CHECKS.append((bool(cond), label))
    print(("ok   " if cond else "FAIL ") + label)


def find_bash4():
    """The block uses arrays and [[ ]]; prefer a modern bash over macOS 3.2."""
    for cand in ("/opt/homebrew/bin/bash", "/usr/local/bin/bash", "/bin/bash"):
        if Path(cand).exists():
            return cand
    return "bash"


BASH = find_bash4()

# Records every invocation. `remote set-url` always succeeds: it is a local
# config write, and the scrub must stay observable on the failure paths.
# Everything else obeys GIT_ERR.
GIT_STUB = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$GIT_LOG"
case "$*" in
    *"remote set-url"*) exit 0 ;;
esac
if [ -n "$GIT_ERR" ]; then
    printf '%s\n' "$GIT_ERR" >&2
    exit 1
fi
if [ "$1" = clone ]; then
    for a in "$@"; do last="$a"; done
    mkdir -p "$last/.git"
fi
exit 0
"""

# The retries would otherwise cost 75s per network case.
SLEEP_STUB = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$SLEEP_LOG"
exit 0
"""


def extract_block() -> str:
    """The client sync block, verbatim, out of the shipped entrypoint."""
    lines = SCRIPT.read_text().splitlines()
    beg = [i for i, l in enumerate(lines) if f"{MARKER}: begin" in l]
    end = [i for i, l in enumerate(lines) if f"{MARKER}: end" in l]
    ok(len(beg) == 1 and len(end) == 1 and beg[0] < end[0],
       f"start_script.sh must carry exactly one '{MARKER}' begin/end marker "
       f"pair (found begin={beg} end={end})")
    return "\n".join(lines[beg[0]:end[0] + 1])


class Run:
    def __init__(self, rc, out, err, git, sleeps, repo_dir):
        self.rc = rc
        self.out = out
        self.err = err
        self.all = out + err
        self.git = git            # every git argv, in order
        self.sleeps = sleeps
        self.repo_dir = repo_dir

    @property
    def syncs(self):
        """clone / fetch attempts, i.e. one per retry."""
        return [l for l in self.git
                if l.startswith("clone ") or " fetch " in f" {l} "]

    @property
    def set_urls(self):
        return [l.split("origin ", 1)[1] for l in self.git
                if "remote set-url" in l]

    @property
    def reached_end(self):
        return "REACHED_END=1" in self.out

    @property
    def pat_after(self):
        for line in self.out.splitlines():
            if line.startswith("GITHUB_PAT_AFTER="):
                return line.split("=", 1)[1]
        return None


BLOCK = None  # filled by main(), after the marker check has run


def run(label, git_err="", token=TOKEN, existing_copy=False):
    """Drive the real block once. Always asserts the token stayed out of the
    output — that constraint holds on every path, so it is checked on every
    path rather than in one dedicated test."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stub_bin = tmp / "bin"
        stub_bin.mkdir()
        for name, body in (("git", GIT_STUB), ("sleep", SLEEP_STUB)):
            p = stub_bin / name
            p.write_text(body)
            p.chmod(0o755)

        repo_dir = tmp / "client"
        if existing_copy:
            (repo_dir / ".git").mkdir(parents=True)

        git_log = tmp / "git.log"
        sleep_log = tmp / "sleep.log"

        script = tmp / "block.sh"
        script.write_text(
            "#!/usr/bin/env bash\n" + BLOCK + "\n"
            # Proves the block reached its end, and reports whether the token
            # survived into the environment the runtime would inherit.
            'printf "GITHUB_PAT_AFTER=%s\\n" "${GITHUB_PAT-<unset>}"\n'
            'printf "REACHED_END=1\\n"\n')

        env = {
            "PATH": f"{stub_bin}:{os.environ.get('PATH', '')}",
            "HOME": str(tmp),
            "CLIENT_REPO_DIR": str(repo_dir),
            "CLIENT_REPO_SLUG": SLUG,
            "CLIENT_REPO_BRANCH": BRANCH,
            "GIT_LOG": str(git_log),
            "SLEEP_LOG": str(sleep_log),
            "GIT_ERR": git_err,
        }
        if token is not None:
            env["GITHUB_PAT"] = token

        r = subprocess.run([BASH, str(script)], env=env,
                           capture_output=True, text=True, timeout=120)
        git = git_log.read_text().splitlines() if git_log.exists() else []
        sleeps = sleep_log.read_text().splitlines() if sleep_log.exists() else []
        res = Run(r.returncode, r.stdout, r.stderr, git, sleeps, repo_dir)

    if token:
        ok(token not in res.all,
           f"[{label}] the token never reaches stdout/stderr")
        ok(token not in "\n".join(res.git) or
           all(token not in u for u in res.set_urls[-1:]),
           f"[{label}] no credentialed URL is left on the remote")
    return res


# --- the error strings, as git actually emits them --------------------------
# The credentialed URL is deliberately embedded in several of these: git echoes
# the remote back in `unable to access '<url>'`, so these are the cases where
# redaction is load-bearing rather than theoretical.

AUTH_CASES = {
    "Authentication failed":
        f"remote: Invalid username or password.\n"
        f"fatal: Authentication failed for '{TOKEN_URL}/'",
    "repository not found":
        "remote: Repository not found.\n"
        f"fatal: repository '{TOKEN_URL}/' not found",
    "http 403":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "The requested URL returned error: 403",
    "http 401":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "The requested URL returned error: 401",
    "http 404":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "The requested URL returned error: 404",
    "could not read Username":
        "fatal: could not read Username for 'https://github.com': "
        "No such device or address",
    "terminal prompts disabled":
        "fatal: could not read Username for 'https://github.com': "
        "terminal prompts disabled",
}

NET_CASES = {
    "dns":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "Could not resolve host: github.com",
    "connect":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "Failed to connect to github.com port 443 after 21 ms: "
        "Couldn't connect to server",
    "timeout":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "Failed to connect to github.com port 443: Connection timed out",
    "tls":
        f"fatal: unable to access '{TOKEN_URL}/': SSL connect error",
    "http 503":
        f"fatal: unable to access '{TOKEN_URL}/': "
        "The requested URL returned error: 503",
}


# --- tests ------------------------------------------------------------------

def test_no_token_fails_before_git_is_touched():
    """A pod with no PAT must say which variable is missing and stop. It must
    NOT reach git: a credential prompt nobody can answer would hang the boot,
    and a tokenless fetch of a private repo is a 404 the operator then has to
    decode."""
    for label, tok in (("unset", None), ("empty string", "")):
        r = run(f"no token/{label}", token=tok)
        ok(r.rc != 0, f"[{label}] no GITHUB_PAT exits non-zero (got {r.rc})")
        ok("GITHUB_PAT" in r.all,
           f"[{label}] the message names the variable: {r.all!r}")
        ok(r.git == [],
           f"[{label}] no git call is attempted: {r.git}")
        ok(not r.reached_end, f"[{label}] the block did not fall through")


def test_happy_path_carries_the_token_then_scrubs_it():
    """Both shapes: a first boot that clones, and a container restart whose
    writable layer still holds the checkout (CLAUDE.md section 5 — never a
    plain clone into an existing dir)."""
    for label, existing in (("fresh clone", False), ("existing checkout", True)):
        r = run(f"happy/{label}", existing_copy=existing)
        ok(r.rc == 0, f"[{label}] a clean sync exits 0: {r.rc} {r.all!r}")
        ok(r.reached_end, f"[{label}] the block runs to its end")

        credentialed = [l for l in r.git if "x-access-token:" in l]
        ok(len(credentialed) == 1,
           f"[{label}] exactly one git call carries the credential: {credentialed}")
        ok(credentialed and TOKEN_URL in credentialed[0],
           f"[{label}] and it is x-access-token:<token>@github.com/<slug>.git: "
           f"{credentialed}")
        if existing:
            ok(credentialed and "remote set-url" in credentialed[0],
               f"[{label}] the existing checkout is re-pointed, not re-cloned: "
               f"{r.git}")
            ok(any(" fetch " in f" {l} " for l in r.git)
               and any(" reset --hard " in f" {l} " for l in r.git),
               f"[{label}] it fetches + hard-resets: {r.git}")
        else:
            ok(credentialed and credentialed[0].startswith("clone "),
               f"[{label}] a missing checkout is cloned: {r.git}")

        ok(r.set_urls and r.set_urls[-1] == CLEAN_URL,
           f"[{label}] the remote is left tokenless: {r.set_urls}")
        ok("✅" in r.out and SLUG in r.out,
           f"[{label}] the boot log names the template it synced: {r.out!r}")


def test_the_token_is_unset_for_every_child_process():
    """AC: absent from the environment of the runtime, ComfyUI and JupyterLab.
    The block's last statement is `unset GITHUB_PAT`, and everything after the
    end marker runs with it gone."""
    r = run("unset", existing_copy=True)
    ok(r.pat_after == "<unset>",
       f"GITHUB_PAT is unset when the block ends (got {r.pat_after!r})")

    tail = SCRIPT.read_text().split(f"{MARKER}: end", 1)[1]
    ok("GITHUB_PAT" not in tail,
       "nothing after the block reads GITHUB_PAT, so unsetting it is safe")


def test_auth_failures_fail_closed():
    """The whole point of the gate: revoking a client's PAT stops their pod.
    Each case runs WITH a usable on-disk copy, so falling back is available and
    must still be refused.

    401/403/404 are the subtle ones — git wraps them in `unable to access`,
    which is also on the network list, so this only passes if the auth list is
    consulted first."""
    for label, err in AUTH_CASES.items():
        r = run(f"auth/{label}", git_err=err, existing_copy=True)
        ok(r.rc != 0, f"[auth/{label}] exits non-zero (got {r.rc})")
        ok(not r.reached_end,
           f"[auth/{label}] the block never falls through to the runtime")
        ok("on-disk copy" not in r.all and "may be stale" not in r.all,
           f"[auth/{label}] the on-disk copy is NOT booted: {r.all!r}")
        ok("✅" not in r.out,
           f"[auth/{label}] no success line is printed: {r.out!r}")
        ok(len(r.syncs) == 1,
           f"[auth/{label}] fails on the first attempt, no retry storm against "
           f"a credential GitHub already rejected: {r.syncs}")
        ok(r.sleeps == [],
           f"[auth/{label}] and it does not sleep: {r.sleeps}")
        ok(r.set_urls and r.set_urls[-1] == CLEAN_URL,
           f"[auth/{label}] the remote is scrubbed on the failure exit too: "
           f"{r.set_urls}")
        # git's own words are echoed so the operator can see what GitHub said —
        # verbatim when there is nothing to hide, redacted when the message
        # carries the credentialed URL back.
        last = err.splitlines()[-1]
        if TOKEN in last:
            ok("<redacted>" in r.all,
               f"[auth/{label}] git's output is shown with the token redacted: "
               f"{r.all!r}")
        else:
            ok(last in r.all,
               f"[auth/{label}] git's output is shown verbatim: {r.all!r}")


def test_auth_failure_message_points_at_the_token():
    """AC: 'exits non-zero with a message naming the token'."""
    r = run("auth/message", git_err=AUTH_CASES["repository not found"],
            existing_copy=True)
    ok("GITHUB_PAT" in r.all,
       f"the auth failure names GITHUB_PAT: {r.all!r}")
    ok("not found" in r.all.lower() and "same cause" in r.all.lower(),
       f"and explains that GitHub reports an invisible private repo as "
       f"'not found', so the operator does not go hunting for a deleted repo: "
       f"{r.all!r}")


def test_network_failures_fail_open_when_a_copy_exists():
    """A GitHub outage must not brick a client pod. Retry, then boot what is on
    disk and say it may be stale. 503 is on this list on purpose: an outage is
    not a credential problem, even though git wraps it in the same
    `unable to access` prefix as a 403."""
    for label, err in NET_CASES.items():
        r = run(f"net/{label}", git_err=err, existing_copy=True)
        ok(r.rc == 0, f"[net/{label}] boots anyway (got rc={r.rc}) {r.all!r}")
        ok(r.reached_end, f"[net/{label}] the block falls through to the runtime")
        ok("may be stale" in r.all,
           f"[net/{label}] and says the copy may be stale: {r.all!r}")
        ok(len(r.syncs) == 5,
           f"[net/{label}] it retried before giving up: {r.syncs}")
        ok(len(r.sleeps) >= 4,
           f"[net/{label}] with a backoff between attempts: {r.sleeps}")
        ok(r.set_urls and r.set_urls[-1] == CLEAN_URL,
           f"[net/{label}] the remote is scrubbed on the fallback path: "
           f"{r.set_urls}")
        ok(r.pat_after == "<unset>",
           f"[net/{label}] and the token is still unset afterwards")


def test_network_failure_with_no_copy_aborts():
    """Nothing to fall back to. Failing open here would exec the runtime
    against a directory that does not exist."""
    for label, err in NET_CASES.items():
        r = run(f"net-nocopy/{label}", git_err=err, existing_copy=False)
        ok(r.rc != 0, f"[net-nocopy/{label}] exits non-zero (got {r.rc})")
        ok("no local copy" in r.all,
           f"[net-nocopy/{label}] and says why: {r.all!r}")
        ok(not r.reached_end,
           f"[net-nocopy/{label}] the block does not fall through")


def test_unknown_git_output_fails_closed():
    """`unknown` is treated exactly like `auth`. Defaulting to closed is the
    point: an error nobody anticipated is far more likely to be a dead token
    than a new dialect of DNS failure."""
    weird = "fatal: the remote end hung up unexpectedly"
    r = run("unknown/text", git_err=weird, existing_copy=True)
    ok(r.rc != 0, f"[unknown] exits non-zero (got {r.rc})")
    ok(not r.reached_end, "[unknown] the block does not fall through")
    ok("on-disk copy" not in r.all,
       f"[unknown] the on-disk copy is not booted: {r.all!r}")
    ok("matches no known network failure" in r.all,
       f"[unknown] the message says the classification failed: {r.all!r}")
    ok(weird in r.all,
       f"[unknown] and prints git's output verbatim, so a misclassification is "
       f"diagnosable in one look instead of one boot at a time: {r.all!r}")
    ok(r.set_urls and r.set_urls[-1] == CLEAN_URL,
       f"[unknown] the remote is scrubbed on this exit path too: {r.set_urls}")


def test_empty_git_output_blames_the_container_not_the_token():
    """git exiting non-zero with nothing on stderr is a container problem —
    OOM kill, disk full, git missing. Without this branch the operator gets a
    blank line and no diagnosis."""
    # An empty GIT_ERR makes the shared stub SUCCEED, so this case needs its
    # own git: one that exits 128 having said nothing at all.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stub_bin = tmp / "bin"
        stub_bin.mkdir()
        g = stub_bin / "git"
        g.write_text('#!/usr/bin/env bash\n'
                     'printf \'%s\\n\' "$*" >> "$GIT_LOG"\n'
                     'case "$*" in *"remote set-url"*) exit 0 ;; esac\n'
                     'exit 128\n')
        g.chmod(0o755)
        s = stub_bin / "sleep"
        s.write_text(SLEEP_STUB)
        s.chmod(0o755)
        repo_dir = tmp / "client"
        (repo_dir / ".git").mkdir(parents=True)
        script = tmp / "block.sh"
        script.write_text("#!/usr/bin/env bash\n" + BLOCK + "\n"
                          'printf "REACHED_END=1\\n"\n')
        env = {
            "PATH": f"{stub_bin}:{os.environ.get('PATH', '')}",
            "HOME": str(tmp),
            "CLIENT_REPO_DIR": str(repo_dir),
            "CLIENT_REPO_SLUG": SLUG,
            "CLIENT_REPO_BRANCH": BRANCH,
            "GIT_LOG": str(tmp / "git.log"),
            "SLEEP_LOG": str(tmp / "sleep.log"),
            "GITHUB_PAT": TOKEN,
        }
        p = subprocess.run([BASH, str(script)], env=env,
                           capture_output=True, text=True, timeout=120)
        both = p.stdout + p.stderr

    ok(p.returncode != 0,
       f"[silent git] exits non-zero (got {p.returncode})")
    ok("REACHED_END=1" not in p.stdout,
       "[silent git] the block does not fall through")
    ok("printed NOTHING" in both and "Suspect the container" in both,
       f"[silent git] the message distinguishes an empty error from an unknown "
       f"one: {both!r}")
    ok("on-disk copy" not in both,
       f"[silent git] the on-disk copy is not booted: {both!r}")
    ok(TOKEN not in both, f"[silent git] the token never reaches output")


def test_the_credentialed_url_is_redacted_out_of_every_printed_error():
    """git echoes the remote back in `unable to access '<url>'`, so every path
    that prints git's output is printing a URL with the token in it unless
    client_redact does its job. This is the constraint that keeps the token out
    of $NETWORK_VOLUME/comfyui.log."""
    for label, err in list(AUTH_CASES.items()) + list(NET_CASES.items()):
        if TOKEN not in err:
            continue
        r = run(f"redact/{label}", git_err=err, existing_copy=True)
        ok(TOKEN not in r.all, f"[redact/{label}] no token: {r.all!r}")
        ok(TOKEN[:12] not in r.all,
           f"[redact/{label}] not even a partial: {r.all!r}")


def test_the_block_is_self_contained():
    """The contract that makes this test possible, and the contract that keeps
    the gate readable: everything the block needs is defined inside it."""
    text = SCRIPT.read_text()
    body = extract_block()
    ok(body.rstrip().endswith("# --- client sync: end"),
       "the block ends at its own marker")
    ok("unset GITHUB_PAT" in body.splitlines()[-2],
       f"the last statement in the block is `unset GITHUB_PAT`: "
       f"{body.splitlines()[-2]!r}")
    ok(text.index(f"{MARKER}: begin") < text.index("RUNTIME_URL"),
       "the client sync runs before the runtime clone, so a rejected token "
       "never costs a public clone")
    ok("x-access-token" not in text.split(f"{MARKER}: end", 1)[1],
       "no credential is used on the public runtime clone")


def main():
    global BLOCK
    if not SKELETON.is_dir():
        print(f"client entrypoint self-test: SKIPPED everything "
              f"(sibling repo not checked out at {SKELETON})")
        print("client entrypoint self-test: WARNING the private-client access "
              "gate was NOT exercised here — check it out beside "
              "comfyui-runtime to run this suite")
        print("\n0 checks run, 1 skipped")
        return 0
    ok(SCRIPT.is_file(), f"{SCRIPT} exists")
    if not SCRIPT.is_file():
        print(f"\n0/{len(CHECKS)} checks passed")
        return 1
    BLOCK = extract_block()

    for t in (test_no_token_fails_before_git_is_touched,
              test_happy_path_carries_the_token_then_scrubs_it,
              test_the_token_is_unset_for_every_child_process,
              test_auth_failures_fail_closed,
              test_auth_failure_message_points_at_the_token,
              test_network_failures_fail_open_when_a_copy_exists,
              test_network_failure_with_no_copy_aborts,
              test_unknown_git_output_fails_closed,
              test_empty_git_output_blames_the_container_not_the_token,
              test_the_credentialed_url_is_redacted_out_of_every_printed_error,
              test_the_block_is_self_contained):
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
