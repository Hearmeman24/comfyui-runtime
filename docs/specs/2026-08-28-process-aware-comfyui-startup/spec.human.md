# Keep slow ComfyUI boots in “still starting” state

**Type:** `bug-fix/recon` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done

A healthy but slow ComfyUI process will keep reporting that it is still starting after 70 seconds,
with the latest meaningful startup-log line. The runtime will poll port 8188 every 5 seconds and
only print the failed-start report if the launched ComfyUI process exits before the port opens.

## 🪤 Gotchas

- Custom nodes can log recoverable import errors while ComfyUI continues to a usable server, so a
  text match for `error` or `Traceback` cannot be the failure authority. Process exit is the fatal
  signal; the log supplies progress and failure diagnostics.
- The container must still remain alive after a ComfyUI failure so the customer can inspect logs.
- This is a tier-1 Runtime change. Merging it to `main` does not deploy it; moving `stable` would
  reach all active templates on their next boot.

## Done when

- [x] A live ComfyUI process is never labelled failed solely because startup crossed 70 seconds.
- [x] Port 8188 is polled at 5-second intervals.
- [x] Slow-start messages include the latest non-empty ComfyUI log line.
- [x] A ComfyUI process that exits before readiness is labelled failed immediately and its log tail
  plus the existing troubleshooting guidance are printed.
- [x] The real shell block and the boot-report failure header remain covered by executable tests.

## The plan

1. Capture the PID and log path from the one permitted ComfyUI launch.
2. Replace the elapsed-time failure with a process-aware 5-second liveness loop and rate-limited
   progress output.
3. Add an exact-shell regression harness for ready, slow-but-alive, and exited-process paths.
4. Update the runtime contract and run the focused and full CI-equivalent verification suites.
