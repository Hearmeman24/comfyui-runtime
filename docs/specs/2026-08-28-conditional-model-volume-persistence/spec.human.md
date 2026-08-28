# Make model persistence optional

**Type:** `feature/app` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done

Models staged on local NVMe will still be copied to the network volume by default. Setting
`PERSIST_MODELS_TO_VOLUME=false` will keep those staged models local and usable for the current pod
without starting the background volume copy.

## 🪤 Gotchas

- This controls the background copy, not staging. `HF_STAGE_LOCAL` keeps its current internal
  meaning.
- A model that cannot fit in local staging may still download directly to the network volume.
- With no network volume, the setting has no effect; models already land on the container disk.
- Locally staged models left unpersisted must download again after the container disk is discarded.

## Done when

- [ ] With the variable unset, pending staged models start the same background copy as today.
- [ ] A trimmed, case-insensitive literal `false` skips that copy and logs the non-durable state.
- [ ] Other values do not silently disable persistence.
- [ ] Existing no-volume, direct-to-volume fallback, and restart/refetch behavior remain unchanged.
- [ ] The runtime contract documents the new variable and its interaction with local staging.

## The plan

1. Add an opt-out gate around the existing background `volume_sync.py` launch.
2. Exercise the shipped shell block with pending, empty, disabled, and no-volume states.
3. Document the persistence contract and run the runtime's focused and full verification.
