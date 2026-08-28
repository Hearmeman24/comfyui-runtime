# Keep slow ComfyUI boots in “still starting” state

- **Work type:** `bug-fix/recon`
- **Status:** `implemented locally`; no material decision is unresolved
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context

The runtime currently reports a failed start when ComfyUI has not answered on port 8188 after 70
seconds, even if the launched Python process is healthy and is still completing startup. On the
reported H200 pod, ComfyUI opened port 8188 about 74.5 seconds after launch, so the fixed deadline
produced a false failure roughly 4.5 seconds before the server became ready.

## 2. Root Cause / Mechanism

- Cause: the liveness loop has a hard `max_wait=70`, breaks at that elapsed time, and the following
  probe records `ready=false` without checking whether the launched ComfyUI process is still alive
  — evidence: `src/start.sh:869-914`.
- Cause: the launch deliberately does not capture `$!`, so the runtime cannot distinguish a slow
  process from one that exited — evidence: `src/start.sh:860-867`.
- Cause: the deployment renderer maps every `ready=false` verdict to `ComfyUI FAILED to start` —
  evidence: `src/boot_report.py:272-283`.
- Confirmed by repro: yes. The live pod's ComfyUI process remained alive and localhost port 8188
  returned HTTP 200 after the runtime had already printed the 70-second failed-start report.

## 3. Acceptance Criteria

- [x] A launched ComfyUI process that remains alive past 70 seconds stays under observation and is
  reported as still starting, never failed because of elapsed time alone. → (ask: “as long as
  there's no errors print that it's still starting instead of terminating after 70 seconds”)
- [x] The readiness probe runs every 5 seconds. → (ask: “instead of a 1 second poll maybe a 5 second
  poll so it's less intrusive”)
- [x] At and after the slow-start threshold, progress output includes the latest non-empty line from
  the ComfyUI startup log. → (ask: “interpret the ComfyUI log”)
- [x] If the captured ComfyUI PID exits before port 8188 opens, record `ready=false`, print the log
  tail and existing troubleshooting block, and leave the container alive for inspection. → (ask:
  “as long as there's no errors”)
- [x] If port 8188 opens, record `ready=true` and render the normal ready report. → (ask: “still
  starting”)

## 4. Scope & Non-Goals

**In scope:** the launch/liveness block in `src/start.sh:860-914`, its executable regression test,
the failure fixture in `tools/test_boot_report.py:298-306`, and the boot-order contract in
`CONTRACTS.md:748-760`.

**Non-goals (explicitly NOT doing):** restarting ComfyUI; killing a slow process; changing model
downloads or volume persistence; monitoring crashes after readiness; changing the `stable` branch;
classifying fatality by grepping log words; mutating the live pod.

## 5. Key Decisions & Constraints

- **Decided:** PID liveness plus HTTP readiness are authoritative. The latest non-empty log line is
  informational, and the final log tail is diagnostic. This avoids treating recoverable custom-node
  import failures as a dead server.
- **Decided:** retain 70 seconds only as the point where wording changes from “starting” to “still
  starting”; it is no longer a deadline.
- **Decided:** poll every 5 seconds but emit routine progress at most every 30 seconds, plus exactly
  at the 70-second slow threshold, to reduce both probing and log noise.
- **Constraint / must-not-break:** launch ComfyUI exactly once with `nohup`, never pipe that launch
  through `tee`, and keep `sleep infinity` after the report — evidence: `src/start.sh:860-867` and
  `src/start.sh:926-927`.
- **Mirror existing:** execute marked shell blocks under stubbed dependencies as done by
  `tools/test_sage_background.py` and `tools/test_volume_sync_launch.py`.

## 6. Code Surface Map

- `src/start.sh:860-927` — authoritative ComfyUI launch, liveness verdict, diagnostics, and report.
- `src/boot_report.py:272-283` — maps the final `ready` verdict to the report header.
- `tools/test_boot_report.py:298-306` — regression that a genuine not-ready verdict never claims
  success.
- `CONTRACTS.md:676-760` — shared runtime boot-order contract.
- `.circleci/config.yml:140-174` — syntax, shellcheck, byte-compile, and globbed stdlib self-test
  gates that every new `tools/test_*.py` suite must pass.

## 7. Ultracode Dispatch Notes

**Build first (sequential — freezes interfaces before any parallelism):**
- The marked `src/start.sh` liveness block and its `ready=true|false` report-state contract.

**Parallel slices (independent — one agent each):**
- None. The shell behavior, exact-shell harness, failure fixture, and contract describe one coupled
  boot transition and are intentionally serialized.

**⛓ Collision audit:** one local writer owns all in-scope paths; no shared write surface exists.

**Each agent must:** not applicable; this task is executed as one serialized slice.

**Recon questions to parallelize:** none; the live failure mode and code mechanism are reproduced.

```yaml
dispatch:
  frozen:
    - src/hf_download_manager.py
    - src/volume_sync.py
    - src/provisioner.py
  slices:
    - key: liveness
      writes:
        - src/start.sh
        - tools/test_comfyui_liveness.py
        - tools/test_boot_report.py
        - CONTRACTS.md
        - ARCHITECTURE.md
  testRunner: "python3 tools/test_comfyui_liveness.py && python3 tools/test_boot_report.py"
```

## 8. Assumptions & Open Questions

None. The launch, report-state ownership, runtime behavior, and live timing were inspected directly.
