# Make model persistence optional

- **Work type:** `feature/app`
- **Status:** `approved` — the environment-variable name and behavior were resolved in chat
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context

The NVMe-first model path always starts a detached stage-to-volume copy when the manifest contains
a live staged symlink and a network volume exists. The requested operator control is to retain that
behavior by default while allowing an explicit environment variable to leave locally staged models
ephemeral for the current pod.

## 2. Approach & Why

- The runtime receives every RunPod template environment variable, so no template schema or env
  allowlist needs extending — evidence: `CONTRACTS.md:600-615`.
- The existing persistence boundary is the detached `volume_sync.py` launch after provisioning; it
  is already gated on pending symlinks and network-volume presence — evidence: `src/start.sh:750-785`.
- `HF_STAGE_LOCAL` controls stage placement. When false it returns a volume-adjacent staging path,
  which is atomically moved to the destination instead of deferred — evidence:
  `src/hf_download_manager.py:319-378`, `src/hf_download_manager.py:490-517`.
- A live staged symlink is reusable on the same pod, while a dangling link is removed and refetched
  after its local target disappears — evidence: `src/hf_download_manager.py:381-406`.

## 3. Acceptance Criteria

- [ ] With `PERSIST_MODELS_TO_VOLUME` unset, pending locally staged models start the existing
  detached volume copy. → (ask: "true by default")
- [ ] With `PERSIST_MODELS_TO_VOLUME` equal to a trimmed, case-insensitive literal `false`, the
  runtime does not launch `volume_sync.py` and emits a clear log line explaining that the models are
  not durable. → (ask: "unless explicitly turned off by an env var")
- [ ] Values other than literal `false` retain persistence. → (ask: "unless explicitly turned off")
- [ ] `HF_STAGE_LOCAL` retains its existing behavior and name. → (ask: "PERSIST_MODELS_TO_VOLUME it is")
- [ ] With no network volume, no background copy launches regardless of the new variable. → (ask:
  the clarified no-workspace behavior)

## 4. Scope & Non-Goals

**In scope:** `src/start.sh:750-785`, a shell-boundary regression under `tools/`, the runtime env
contract in `CONTRACTS.md:783-794`, and the NVMe-first architecture note.

**Non-goals (explicitly NOT doing):** changing local-stage selection or headroom; preventing the
existing direct-to-volume fallback; changing no-volume downloads; adding a `template.json` field;
promoting `stable` or deploying the runtime.

## 5. Key Decisions & Constraints

- **Decided:** the public variable is `PERSIST_MODELS_TO_VOLUME`, default enabled; only a stripped,
  lowercased literal `false` disables it.
- **Decided:** the gate belongs at the `volume_sync.py` launch, not in the downloader, because stage
  selection and persistence are independent — evidence: `src/start.sh:750-785`,
  `src/hf_download_manager.py:319-378`.
- **Constraint / must-not-break:** disabling persistence leaves the live destination symlink intact
  so ComfyUI can use the local staged model — evidence: `src/volume_sync.py:36-53`.
- **Constraint / must-not-break:** a short or failed copy must never replace the working symlink —
  evidence: `src/volume_sync.py:56-74`, `src/volume_sync.py:96-107`.
- **Scale:** manifest-sized model sets, copied sequentially by the existing detached worker; the
  toggle must add no new per-model work.

## 6. Code Surface Map

- `src/start.sh:750-785` — detects pending staged models and launches the detached volume sync.
- `src/hf_download_manager.py:319-378` — chooses local versus volume-side staging.
- `src/hf_download_manager.py:381-406` — handles live and dangling staged symlinks on later boots.
- `src/volume_sync.py:36-119` — performs the atomic background persistence.
- `tools/test_volume_sync.py:1-194` — validates per-model copy safety and idempotence.
- `CONTRACTS.md:783-794` — authoritative runtime environment-variable table.

## 7. Ultracode Dispatch Notes

**Build first (sequential — freezes interfaces before any parallelism):**

- Freeze `PERSIST_MODELS_TO_VOLUME`: default on; only trimmed, case-insensitive `false` opts out.

**Parallel slices:** none. The shell gate, its extracted-block test, and contract documentation are
one serialized slice because the test must execute the exact block it validates.

**⛓ Collision audit:** one writer owns all changed runtime paths; no parallel write sets exist.

**Each agent must:** implement its slice + write and green its own tests + self-verify against §3.

```yaml
dispatch:
  frozen:
    - src/hf_download_manager.py
    - src/volume_sync.py
  slices:
    - key: persistenceGate
      writes:
        - src/start.sh
        - tools/test_volume_sync_launch.py
        - CONTRACTS.md
        - ARCHITECTURE.md
  testRunner: "python3 tools/test_volume_sync_launch.py"
```

## 8. Assumptions & Open Questions

None. The environment boundary, staging behavior, sync launch, and restart handling were verified
against the current runtime checkout before implementation.
