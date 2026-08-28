# How the templates and this runtime fit together

`CONTRACTS.md` is the reference: every field, every exit code. This file is the map you read first,
to understand *why* the pieces sit where they do and what happens when you change one.

Written 2026-08-14, when the last of the four templates finished migrating.

---

## The shape

There are four active public templates — **wan**, **minimax**, **qwen-image**, **ltx2** — and one
runtime, this repo. The templates share the runtime rather than each carrying a copy.

A template repo now holds only what is genuinely its own:

```
template-repo/
  Dockerfile              the node packs this family needs, on the shared base image
  pins.json               which runtime and which base image this template rides
  template.json           what this template IS, expressed as configuration
  src/start_script.sh     the image-baked entrypoint (~100 lines, near-identical across the four)
  src/start.sh            a 52-line version guard: refuses to boot a pre-migration image
  src/models_registry.json
  src/note_sections.md    the per-template half of the in-ComfyUI welcome note
  src/hooks/              optional pre_download.sh / pre_launch.sh escape hatches
  workflows/**
  tools/                  thin shims that fetch and run this repo's real tools
```

Everything else — downloading, provisioning, the boot script, the boot report, the validator, the
SageAttention build, CivitAI env resolution — lives here and is identical on every pod.

The rule that keeps it that way: **the runtime is a program, `template.json` is its configuration.**
When something cannot be expressed in `template.json`, that is the signal to extend the schema, not
to add bespoke code to one template.

---

## What happens at boot

1. RunPod starts the container. The entrypoint is `src/start_script.sh`, **baked into the image**.
2. It fetches the *template* repo's default branch and hard-resets to it, in a 5-attempt retry loop
   with backoff. On total failure it falls back to the on-disk copy and says so.
3. It reads `pins.json`, clones **this repo** at `runtime_ref`, and copies the runtime scripts in.
   Same retry loop, same fallback.
4. It `exec`s this repo's `src/start.sh`, which owns everything from there: DNS preflight, node
   packs, provisioning, downloads, SageAttention, the boot report, and the ComfyUI launch.

Two consequences worth internalising:

**`start_script.sh` is the expensive file.** It is the only thing that needs a new image to change.
Everything else in a template ships by merging.

**A template's `src/start.sh` is not a boot script any more.** It is a 52-line banner that prints an
upgrade message and sleeps, so a customer still running a pre-migration image gets told to update
rather than silently booting half-migrated.

---

## Three tiers of change

Know which tier you are in before you write anything. They have very different blast radii.

| Tier | What | How it ships | Reaches |
|---|---|---|---|
| **1. Runtime** | anything in this repo | merge to `main` ships **nothing**; promote by moving `stable` | **all four templates at once**, next pod boot |
| **2. Template runtime files** | `src/` (except `start_script.sh`), `template.json`, `pins.json`, `workflows/**` | merge to the template's default branch | that template, next pod boot |
| **3. Image** | `Dockerfile`, `src/start_script.sh` | merge → `vN` git tag → CircleCI → Docker Hub → repoint the RunPod template | that template, once the customer redeploys |

Prefer the lowest tier that solves the problem.

**The tier-2/tier-3 trap.** A tier-2 merge reaches pods that are still running the *old* image. So
removing something the old image depended on breaks those pods until the new tag is live. wan hit
exactly this: emptying `custom_nodes.repos` while the old image had no baked copy of those packs
would have stripped them from every running pod. **Merge and tag together.**

---

## The promotion model

`pins.json` names a **branch**, not a SHA:

```json
{ "runtime_ref": "stable", "base_image": "hearmeman/comfyui-base:cu130-comfy0.32.0-torch2.11.0" }
```

`start_script.sh` does `git fetch origin "$RUNTIME_REF"` then `reset --hard FETCH_HEAD`, which
resolves a branch exactly as it resolves a SHA. So:

- **Promote:** `git push origin origin/main:refs/heads/stable`
- **Roll back:** move `stable` to the previous commit
- **Stage:** merge to `main`, leave `stable` where it is, and nothing is live yet

This is deliberate. The alternative — a SHA per template — means four PRs to ship one runtime fix,
and four chances to forget one. The cost is that there is **no per-template gate**: a bad `stable`
reaches all four on the next boot. So verify against all four templates before promoting, not just
the one you were working in.

---

## `template.json`, the configuration surface

Full field reference is `CONTRACTS.md` §5. The mental model:

**`provisioning_mode`** decides who chooses the models.
- `walk` (wan, minimax, qwen) — copy the workflows for the enabled flags, then read those files and
  download whatever they name. *The workflows decide.* A `flag` field in the registry does nothing.
- `registry` (ltx2) — the registry decides, via its own `flag` / `on_by_default` / `gated` fields.

**`flags`** are the `download_*` env vars a customer sets, and what each pulls in: `folders` of
workflows to copy, `extra_models` that no workflow names, and a `default`. Absent `default` means
`false`.

**`swap_groups`** are precision/quant switches. One env var picks a profile; the runtime both
downloads those files *and* rewrites the loader widgets in the copied workflows, so the graph always
matches what was downloaded.

**`deprecated_flags`** retire a flag while keeping it *accepted*. Setting it prints one line naming
the replacement and downloads nothing, so a customer with the old value saved in their RunPod
template still boots. **Never delete a flag a customer may have saved.**

**`auto_download`** is a *suppression* list, not a download list: models the node packs fetch
themselves, named here so the boot report stops calling them missing.

**`PERSIST_MODELS_TO_VOLUME`** is the operator opt-out for the detached local-stage-to-volume copy.
It defaults on; only a trimmed, case-insensitive literal `false` leaves locally staged models on the
pod's disk. This does not change stage selection: models that cannot fit locally still stage and land
directly on the volume, and deployments without a network volume already land on container disk.

**`custom_nodes.repos`** are the packs *this* template clones at boot. Packs that every template
should have go in `src/runtime_nodes.json` in this repo instead (`CONTRACTS.md` §5e) — one push and
a `stable` promotion reaches all of them, rather than an identical one-line PR per template repo.
Both lists feed the same loop, deduplicated by directory name, and on a collision the template's
entry wins. Keep the runtime list to packs with no dependencies: a `requirements.txt` there is a
pip install on every boot of every pod.

---

## Adding a template

If this is more than a day's work, the runtime has failed its purpose. In order:

1. `Dockerfile` on the shared base, adding only the node packs your workflows actually resolve.
2. `pins.json` — `runtime_ref: "stable"` and the current `base_image`.
3. `template.json` — flags, folders, mode. Read `CONTRACTS.md` §5 first.
4. `src/models_registry.json`, `src/note_sections.md`, `workflows/**`.
5. Copy `src/start_script.sh` and `tools/validate_models.py` verbatim from wan. They are meant to be
   identical; if you need to change one, the change probably belongs here instead.
6. `.circleci/config.yml` copied from wan: `verify` and `validate_models` on every push, both gating
   `build_and_push`, which fires on a `vN` tag only.

---

## Conventions that are not optional

- **Empty LoRA slots use a placeholder**, `Your_Character_LoRA_Here.safetensors`. Since E13 the
  validator *errors* on a loader naming anything else that is not in the registry, so a personal
  model cannot leak into a shipped workflow again.
- **Canonical CivitAI env names** are `civitai_token`, `CIVITAI_LORAS`, `CIVITAI_CHECKPOINTS`.
  `src/civitai_env.sh` keeps every legacy name working and prints a rename notice.
- **Never bake a token into an image.** Customer-supplied only.
- **Never remove a registry entry an env value maps to.** Use `deprecated_flags`.
- **The repo must stay blob-free.** Every pod clones it on every boot, so repo size is boot time.
  Wheels ship as Release assets, fetched by checksum.
