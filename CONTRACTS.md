# CONTRACTS.md - frozen interfaces for the comfyui-runtime fan-out

Status: FROZEN for the step-0 build (spec `docs/specs/2026-08-12-shared-runtime-scaffold/spec.claude.md`, §7).
Six slice agents code against this file. Do not edit it mid-fan-out; a change here means re-briefing every slice.

How to read this file:

- Every claim about how the family behaves today carries a `file:line` citation into the read-only
  template repos checked out alongside this one.
- Where the approved plan (`plan.md` v4) deliberately changes current behavior, the change is listed in
  §12 so nobody "fixes" it back.
- Judgement calls the spec did not settle are marked inline as `> DECISION NEEDED:` for the maintainer to scan.
- No file in any `comfyui-*` template repo is modified by any slice. They are sources to port from.

Repo layout being built (plan.md §2):

```
comfyui-runtime/
  src/hf_download_manager.py   slice A
  src/provisioner.py           slice B
  src/start.sh                 slice C
  src/sage_probe.py            slice C
  base/Dockerfile              slice E
  torch/cu128.txt              written (build-first, this commit)
  torch/cu130.txt              written (build-first, this commit)
  tools/build_sage_wheel.sh    slice E
  tools/BUILD_WHEELS.md        slice E
  tools/validate_models.py     slice D
  tools/test_*.py              each slice owns its own
  .circleci/config.yml         slice F
```

All tests are stdlib-only, self-contained, run as `python3 <path>` (family convention:
`comfyui-minimax/tools/test_provisioner.py:106` runs under `__main__`; ltx2's `build_manifest.py`
carries a `--selftest`). No pytest anywhere.

---

## 1. Download manifest (TSV)

The interface between `provisioner.py` (writer) and `hf_download_manager.py` (reader). Default path
`/tmp/hf_download_queue.tsv` (`comfyui-wan/src/start.sh:236`), but both tools take the path as an
argument and must not hardcode it.

One entry per line, tab-separated:

```
<url>\t<abs_dest>[\t<min_size_mb>]
```

- `url`: the download URL, verbatim from the registry.
- `abs_dest`: the absolute final path of the file, directories included
  (`comfyui-wan/src/workflow_provisioner.py:111,115`: `models_root / subdir / basename`). The
  downloader creates parent directories.
- `min_size_mb` (optional third field, NEW): a number (float allowed), the "looks complete" floor for
  this file in MB. Absent means 10 MB, today's flat floor
  (`comfyui-wan/src/workflow_provisioner.py:106`, `comfyui-ltx2/src/build_manifest.py:39`). The
  provisioner emits it for registry entries carrying `min_size_mb`
  (`comfyui-ltx2/src/build_manifest.py:92`).

> DECISION NEEDED: the third column is a new, backwards-compatible extension of the two-field format.
> Today `min_size_mb` is honoured only at manifest-build time (`build_manifest.py:92`); slice A's
> brief ("per-entry min_size_mb" in the downloader, spec §7) requires the floor to reach the
> downloader, and a third TSV field is the smallest way. Two-field lines stay valid forever.

Line handling, frozen from `comfyui-wan/src/hf_download_manager.py:45-66`:

- Lines are stripped; blank lines and lines starting with `#` are skipped silently.
- A non-blank line with no tab is skipped with a printed warning, never a fatal error.
- A line with more than the defined fields: fields beyond the third are ignored with a warning.

URL routing (CHANGED from today, see §12):

- An HF resolve URL, matching `^https?://huggingface\.co/<repo_id>/resolve/<rev>/<file>[?...]$`
  (regex frozen at `comfyui-wan/src/hf_download_manager.py:25-27`), goes through `hf_hub_download`
  with hf_xet, staged then atomically moved (see §2).
- A Google Drive URL (host `drive.google.com` or `docs.google.com`) goes through `gdown` (family
  rule, `CLAUDE.md` §3: "aria2c, or gdown for Google Drive"; gdown is installed in the base,
  donor `comfyui-wan/Dockerfile:61`).
- Any other http(s) URL goes through aria2c with qwen's flags:
  `aria2c -x 16 -s 16 -k 1M --continue=true --summary-interval=0 --console-log-level=warn -d <staging> -o <name> <url>`
  (`comfyui-qwen-image/src/download_models.py:104-113`).
- Nothing is ever silently dropped. Today wan's manager prints "skip non-HF url" and drops the entry
  (`comfyui-wan/src/hf_download_manager.py:56-58`); that behavior dies. After the mirror repo
  (plan D3) every SHIPPED model is an HF resolve URL anyway; the direct path exists solely for
  client-private assets (presigned R2, Google Drive) that can never enter a public repo (plan D3a).

A "direct-download entry" therefore needs no special syntax: it is an ordinary registry entry / an
ordinary manifest line whose `url` is not an HF resolve URL. The dispatch is by URL shape, not by a
flag.

---

## 2. src/hf_download_manager.py (slice A)

CLI, frozen from `comfyui-wan/src/hf_download_manager.py:186-192`:

```
python3 hf_download_manager.py <manifest-path>
```

No other arguments. Configuration is by env only.

Exit codes:

| code | meaning | source |
|---|---|---|
| 0 | all entries done, or manifest missing, or manifest empty | wan `:192,196,222` |
| 1 | one or more entries failed, including a watchdog abandon | wan `:220`, ltx2 `:252` (`os._exit(1)`) |
| 2 | usage error (wrong argc) | wan `:189` |

Frozen behavior:

- `POOL_SIZE = 3`. Never raise it (`comfyui-wan/src/hf_download_manager.py:21`, `CLAUDE.md` §3:
  RunPod NFS aggregate caps around 150 MB/s).
- Snapshot logger: append-only status block every 10 s (`SNAPSHOT_INTERVAL`, wan `:22,144-183`).
  Non-TTY safe; no carriage-return progress bars.
- HF timeout env defaults set before `huggingface_hub` import:
  `HF_HUB_DOWNLOAD_TIMEOUT=30`, `HF_HUB_ETAG_TIMEOUT=15` (`comfyui-ltx2/src/hf_download_manager.py:21-22`).
- Watchdog: `STALL_SECS = 300`, `DEADLINE_SECS = 3600` (`comfyui-ltx2/src/hf_download_manager.py:28-29`).
  No global byte progress for STALL_SECS, or phase wall-time past DEADLINE_SECS: mark remaining jobs
  failed, print one line, final snapshot, `os._exit(1)` (ltx2 `:215-252`; stuck `hf_hub_download`
  threads cannot be cancelled, so the pool join is bypassed on purpose). This makes ltx2's outer
  `timeout 4000` wrapper (`comfyui-ltx2/src/start.sh:250`) redundant; it is not ported.
- HF token: passed explicitly, `token=os.environ.get("HF_TOKEN") or None`
  (`comfyui-qwen-image/src/download_models.py:88-96`). Never bake or default a token.
- Skip-if-present: dest exists and `size >= floor` (third manifest field, else 10 MB): count as done,
  print skip line. A dest BELOW the floor is deleted and refetched
  (`comfyui-qwen-image/src/download_models.py:54-61`).
- Post-download verify: final file must exist and be `>= floor`, else the entry is failed.
- Staging and atomic handoff (nicolefire pattern, per plan D3 and spec §7 slice A): download into a
  staging dir on the pod's LOCAL disk (`/hf_stage`) when free space is at least 2.5x the file size;
  otherwise fall back to a volume-side staging dir beside the dest (today's
  `dest.parent/.hf_stage/<name>`, `comfyui-wan/src/hf_download_manager.py:116-119`). Hand off as
  `<dest>.partial` then `os.replace` onto `dest`. A partial file must never be visible at `dest`.
  Never let `hf_hub_download` touch `~/.cache/huggingface` (`CLAUDE.md` §3).
- The downloader never deletes any existing volume file except a sub-floor dest it is about to
  refetch (plan §7 blast radius: "The downloader never deletes existing volume files").

---

## 3. src/provisioner.py (slice B)

One tool replacing three: `comfyui-wan/src/workflow_provisioner.py` (walk),
`comfyui-minimax/src/workflow_provisioner.py` (walk + quant), `comfyui-ltx2/src/build_manifest.py`
(registry), `comfyui-qwen-image/src/provision_models.py` (per-flag lists + precision swap).

CLI:

```
python3 provisioner.py \
    --template  <path>/template.json \
    --registry  <path>/models_registry.json \
    --workflows-src <dir> \
    --workflows-dst <dir> \
    --models-root   <dir> \
    --manifest      <path>

python3 provisioner.py --selftest
```

There is NO `--flag` and NO `--quant` argument. Flag state, quant/precision choice and variant choice
are read from the process environment, mapped through `template.json`. This is forced by the design:
`start.sh` is shared and cannot know per-template flag names, so the per-template shell blocks that
translate env to argv today (`comfyui-wan/src/start.sh:237-242`,
`comfyui-minimax/src/start.sh:205-208`) cannot survive. ltx2's `build_manifest.py` already works this
way (`comfyui-ltx2/src/build_manifest.py:218-222` reads `os.environ`).

Exit codes:

| code | meaning |
|---|---|
| 0 | success. Includes "no flags enabled": write an EMPTY manifest, copy nothing, print one line (`comfyui-wan/src/workflow_provisioner.py:63-66`) |
| 2 | config/usage error: unreadable or invalid `template.json`, unknown `provisioning_mode`, a swap-profile filename missing from the registry (today a raw KeyError at `comfyui-minimax/src/workflow_provisioner.py:101`), a flag entry naming a missing folder is NOT this (warning only, wan `:79-81`) |
| 1 | unexpected runtime error |

What it writes:

1. The manifest TSV at `--manifest` (format §1), always written, possibly empty. Entries already on
   disk at/above their floor are not queued (wan `:106-116`, `build_manifest.py:84-97`); the skip
   floor is `min_size_mb` when the registry entry has it, else 10 MB.
2. The selected workflow JSONs into `--workflows-dst`, preserving each file's path relative to
   `--workflows-src` (`comfyui-wan/src/workflow_provisioner.py:118-125`), in BOTH modes.

> DECISION NEEDED: ltx2 today flattens copies into the workflows root by basename
> (`comfyui-ltx2/src/start.sh:365-375`). Preserving relative paths ("Group shipped workflows under
> folders", `CLAUDE.md` §13) means an existing ltx2 volume gains folderized copies beside its old
> flat ones on the first migrated boot. Duplication, not breakage, and no volume file is deleted.
> Confirm this is acceptable for ltx2's migration (a step-3 concern, but the contract fixes it now).

### Mode: walk (`"provisioning_mode": "walk"`; wan, minimax, qwen)

Per enabled flag, gather workflow files from the flag's `folders` (recursive `*.json`,
wan `:77-85`) or its explicit `workflows` list (qwen's shape,
`comfyui-qwen-image/src/provision_models.py:92-102`). Then:

1. **Rewrite pass** (only when `swap_groups` exist): parse each workflow as JSON and apply the swap
   map (see §5a) to every node's `widgets_values` in `nodes` AND in
   `definitions.subgraphs[].nodes`, and to every `properties.models[]` entry (both `name` and `url`,
   url from the registry) (`comfyui-minimax/src/workflow_provisioner.py:141-160`). Write with
   `json.dumps(doc, indent=2, ensure_ascii=False)`. Without swap groups, copy bytes verbatim
   (`shutil.copy2`, wan `:124`).
2. **Scan pass**: regex model basenames out of the POST-rewrite content with
   `MODEL_PAT = r'"([^"]+\.(?:safetensors|bin|onnx|pth|ckpt))"'` (wan `:44`; the union with qwen's
   extensions adds `pt` and `gguf`: `comfyui-qwen-image/src/provision_models.py:30`). Take
   `os.path.basename` of each hit.
3. **Resolve**: each basename found in the registry is queued; basenames in the template's
   `auto_download` or `image_baked` lists are skipped (wan `:33-42,90-91`); refs equal to ANY swap
   profile filename are never queued directly (managed set, minimax `:51,104-105`); anything else is
   a user-supplied warning, never an error (wan `:131-136`).
4. **Swap-group queue**: for each swap group with an enabled flag, queue every file of the selected
   profile via the registry (minimax `:100-101`), regardless of scan results.
5. **Sidecars**: any registry entry whose `auto_include_with` trigger is queued is added
   (wan `:97-102`).
6. Per-flag `extra_models` are queued via the registry; a missing name prints an error line but does
   not fail the run (`comfyui-qwen-image/src/provision_models.py:119-127`).

### Mode: registry (`"provisioning_mode": "registry"`; ltx2)

Model selection is `build_manifest.py`'s `select()` VERBATIM, including its selftest cases
(`comfyui-ltx2/src/build_manifest.py:42-81,100-207`): flag / `on_by_default` / `disable_flag` /
`gated` / `variant` gating over the whole registry, then the skip-existing floor. The fp8-vs-full
`variant` choice is driven by the env var named in `template.json`'s `variant_env`, resolved through
that env's swap group with `resolve_profile_key` when one exists (fp8 iff the resolved profile key is
`"true"`), and by a stripped, lowercased compare against `"true"` when it does not. Resolving it the
same way the workflow rewrite does is what keeps the queued file and the loader widget on the same
variant (ltx2 hardcodes `lightweight_fp8`, `build_manifest.py:221-222`).

Workflow copying comes from the flag map's `copy` lists (ltx2 does this in bash today,
`comfyui-ltx2/src/start.sh:378-403`): each entry is a path relative to `--workflows-src`; a directory
means all `*.json` under it recursively; `"."` means top-level `*.json` files only, non-recursive.
Rewrite/scan passes run exactly as in walk mode (this replaces ltx2's sed rewrite,
`comfyui-ltx2/src/start.sh:408-417`), but step 3's queueing is informational only in registry mode:
the registry gating decides downloads; the scan exists to produce user-supplied warnings.

> DECISION NEEDED: ltx2 today copies the LTX-2.5 workflows when the USER asked for them, even after
> the token preflight forced `download_ltx25=false`, so the customer sees red nodes plus an
> explanatory notice instead of silently missing workflows (`comfyui-ltx2/src/start.sh:214,400-403`).
> Under this contract, a hook flipping the flag off also drops the workflow copies. If the old
> behavior matters, ltx2's `pre_download.sh` hook can copy them itself; the provisioner will not
> special-case it.

### Flag truthiness (both modes)

- Opt-in flag (`"default": false` or absent): enabled iff `env.strip().lower()` is in
  `{"1","true","yes","on"}` (qwen `env_bool`, `provision_models.py:46-50`; wan/minimax accept only
  `"true"` today, `comfyui-wan/src/start.sh:239`; the union is deliberate and strictly wider).
- Opt-out flag (`"default": true`, ltx2's `on_by_default`): disabled iff `env.strip().lower()` is
  exactly `"false"`; any other value, typos included, keeps it on
  (`comfyui-ltx2/src/build_manifest.py:59-68`).
- `disable_flag` (registry entries): triggered iff the env value is exactly `"true"`
  (`build_manifest.py:70-73`).

---

## 4. models_registry.json schema

One flat JSON object: `basename -> entry`. The key is the exact filename that appears in workflow
widget values and becomes the dest filename. Schema is ltx2's, family-wide (plan D5).

| field | type | required | meaning |
|---|---|---|---|
| `url` | string | yes | Download URL. After the mirror lands (plan D3), every shipped model is an HF resolve URL (originals or `Hearmeman/comfyui-template-assets`); non-HF URLs are legal but reserved for client-private assets. |
| `subdir` | string | yes | Dest dir relative to the models root: `dest = <models_root>/<subdir>/<basename>` (`comfyui-ltx2/src/build_manifest.py:88`). May nest (`"loras/ltx2"`, `comfyui-ltx2/src/models_registry.json:25`). |
| `flag` | string | no | Env-var gate. Absent = always selected (registry mode) / selected by workflow reference (walk mode). |
| `on_by_default` | bool | no | Only meaningful with `flag`: inverts it to opt-out; only a literal `"false"` drops the entry (`build_manifest.py:59-68`). |
| `disable_flag` | string | no | Independent opt-out: entry ships unless that env var is exactly `"true"` (`build_manifest.py:73`). |
| `variant` | string | no | `"full"` or `"fp8"`. Among variant-tagged entries, keep fp8 when the `variant_env` resolves to the `"true"` profile of its swap group (or, with no such group, equals `"true"` after strip and lower), else full. Untagged entries unaffected (`build_manifest.py:77-81`). |
| `gated` | bool | no | HF-gated repo: selected only when `HF_TOKEN` is set and non-blank; without a token the entry is dropped, fail-open (`build_manifest.py:57,74`, selftest `:147-168`). |
| `min_size_mb` | number | no | Overrides the 10 MB "looks complete" floor, both at provision time (`build_manifest.py:92`) and, via the manifest third field, at download time (§1). |
| `auto_include_with` | string | no | Sidecar trigger: entry is added whenever the named basename is queued (`comfyui-wan/src/workflow_provisioner.py:97-102`). Runs after all gating; `baked` still excludes. |
| `baked` | bool | no | File is baked into the image: never queued, and workflow references to it do not warn (`comfyui-qwen-image/src/provision_models.py:107-108`; kept per plan D5). |
| `upstream_url` | string | no | NEW, informational only (plan D3a): the original upstream URL for a file now served from the mirror. No code reads it. |

**`dest_subdir` is DEAD.** It is qwen's private spelling with a `models/` prefix
(`comfyui-qwen-image/src/models_registry.json:4`); plan D5 renames it to `subdir` (relative to the
models root, no prefix) in qwen's migration commit. No runtime code may read or write `dest_subdir`,
including compatibility shims. The validator (slice D) rejects it.

Precedence when fields interact, in evaluation order: `flag`/`on_by_default` -> `disable_flag` ->
`gated` -> `variant` -> `auto_include_with` additions -> `baked` exclusion -> skip-existing floor.
`flag` + `gated` compose with AND: both must pass (`build_manifest.py` selftest `:156-168`).

---

## 5. template.json schema

Per-template config, at the template repo root, consumed by `provisioner.py` and `start.sh`
(plan §2 "What stays per-template": "repo name and branch, provisioning mode, flag map,
variant/precision swap maps, extra extra_model_paths categories, boot-time node clone list with
pins, sage on/off").

```jsonc
{
  "template_repo": "https://github.com/Hearmeman24/comfyui-minimax.git",  // clone URL
  "branch": "master",                    // the branch the entrypoint hard-resets to
  "provisioning_mode": "walk",           // "walk" | "registry"
  "flags": { /* §5b */ },
  "swap_groups": [ /* §5a */ ],          // optional
  "variant_env": "lightweight_fp8",      // optional; registry-mode fp8/full switch (§3)
  "deprecated_flags": {                  // optional; retired flags, still ACCEPTED
    "download_ltx2_19b": "LTX-2 19B is retired; use download_ltx25 instead."
  },                                     //   announced when set, enables nothing, queues nothing
  "auto_download": [ "rife49.pth" ],     // optional; walk mode: fetched by node packs at runtime
                                         //   (comfyui-wan/src/workflow_provisioner.py:33-39)
  "image_baked": [ "4xLSDIR.pth" ],      // optional; walk mode: baked into the image (wan :42)
  "extra_model_paths": [ "vae_approx" ], // optional; categories ADDED to the derived list
                                         //   (2026-08-13 amendment below; native categories
                                         //   no longer need it)
  "models_symlink": false,               // optional, default false; qwen symlinks models/ too
                                         //   (comfyui-qwen-image/src/start.sh:58; plan §4 keeps it
                                         //   as an option so existing volumes see zero change)
  "custom_nodes": {                      // optional; boot-time clone list. THIS
                                         //   template's packs; the runtime adds
                                         //   its own on top (§5e)
    "target": "image",                   // "image" ($COMFYUI_DIR/custom_nodes, wan) |
                                         // "volume" ($PERSIST_ROOT/custom_nodes, qwen :46,109-145)
    "repos": [
      "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
      "https://github.com/kijai/ComfyUI-KJNodes.git|204f6d5",
      "https://github.com/spacepxl/ComfyUI-VAE-Utils.git|force"
    ]
  },
  "sage": true,                          // false skips the whole sage phase: no install, no probe
  "jupyter": true                        // optional, default TRUE; only a literal false skips the
                                         //   JupyterLab launch entirely (`src/start.sh` :185,:200,
                                         //   the JUPYTER-LAUNCH block). Opt OUT, unlike "sage":
                                         //   the four public templates carry no such key and must
                                         //   keep launching it. For private client pods that
                                         //   publish 8188 only — leaving 8888 off the RunPod
                                         //   template hides the proxy route but still runs and
                                         //   binds the process.
}
```

Clone-list entry syntax, frozen from `comfyui-wan/src/start.sh:152-179` plus qwen's `force` mode
(`comfyui-qwen-image/src/start.sh:111-131`): `"<url>"`, `"<url>|<sha>"`, or `"<url>|force"`.
Unpinned: clone if missing, else best-effort `git pull` (`--ff-only`, failures never block boot).
Pinned sha: clone/fetch then `reset --hard <sha>`. `force`: delete the dir first, then clone. After
clone/update, if `requirements.txt` exists, `pip install -r` it (PIP_CONSTRAINT applies
automatically, base-owned; plan §5b "Torch safety").

The `extra_model_paths` base list is wan's, frozen (`comfyui-wan/src/start.sh:120-138`):
checkpoints, clip, clip_vision, controlnet, diffusion_models, embeddings, loras, style_models,
text_encoders, unet, upscale_models, latent_upscale_models, detection, vae, custom_nodes.
Template additions today: minimax `vae_approx` (`comfyui-minimax/src/start.sh:143`), ltx2
`model_patches` (`comfyui-ltx2/src/start.sh:119`), qwen `configs, diffusers, gligen, hypernetworks,
photomaker, vae_approx` (`comfyui-qwen-image/src/start.sh:66-84`). `start.sh` MUST `mkdir -p` every
category it writes into the yaml (`CLAUDE.md` §9: "Generate the yaml from the same list you mkdir").
Removal of a base category is not supported; an unused extra directory is harmless.

**Amended 2026-08-13** (post-step-0; ltx2 migration spec D5: one robust extra_model_paths
across all templates, covering every ComfyUI models subdir). The frozen wan
base list above is superseded: `start.sh` now derives the category list at boot from the pinned
ComfyUI tree's own `folder_paths.folder_names_and_paths` (`src/model_paths.py`), so the list can
never again drift from the ComfyUI the image actually runs. Every registered dir under
`models_dir` is emitted under its canonical key, which carries the legacy alternate dirs along:
`clip` under `text_encoders`, `unet` under `diffusion_models`, `t2i_adapter` under `controlnet`
(`t2i_adapter` is NOT in `map_legacy` and must never be a yaml key of its own). `custom_nodes` and
`datasets` hang off `base_path`, are excluded from the models emission, and `custom_nodes` keeps
its bespoke base_path-relative yaml line. The derivation runs AFTER the `COMFYUI_VERSION` phase,
so the categories come from the tree that will actually run. On any derivation failure
`model_paths.py` prints a frozen v0.32.0 superset (28 dirs) instead and the boot report warns; the
list is never empty and boot never aborts. The yaml is still written from the SAME list that is
`mkdir -p`'d, and the `template.json` `extra_model_paths` key above stays accepted and additive
(node packs that read their own dirs outside `folder_paths`); with every native category derived,
no template should need it. Enforced by `tools/test_model_paths.py`. The step-0 freeze in this
file's header was scoped to that fan-out, which is complete.

### 5a. swap_groups (generalises minimax quant + qwen precision + ltx2's 19b sed)

```jsonc
{
  "env": "minimax_quant",            // the env var the customer sets
  "default": "int8",                 // used when env is unset; unknown values warn and
                                     //   fall back to default (qwen resolve_precision,
                                     //   provision_models.py:65-70), never a hard error:
                                     //   a customer typo must not kill a boot
  "flags": ["download_minimax_h3"],  // group is ACTIVE iff any listed flag is enabled
  "profiles": {                      // profile key == env value; each maps role -> filename
    "int8":  { "fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", ... },
    "fp8":   { "fl2va": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",  ... },
    "nvfp4": { ... }                 // aliasing profiles just repeat the same filenames
                                     //   (minimax workflow_provisioner.py:45-49)
  }
}
```

Frozen semantics, generalised 1:1 from `comfyui-minimax/src/workflow_provisioner.py:34-51,100-105,
126-160`:

- **Selected profile** S = `profiles[env value or default]`.
- **Managed set** = every filename in every profile of every group. Scan-pass references to a managed
  filename never queue directly (minimax `:104-105`; the "Model Links" note prose lists every
  variant, and must keep doing so without triggering downloads).
- **Rewrite map** = for each role r and each profile P != S: `P[r] -> S[r]`. Applied to loader
  widgets AND `properties.models` in every copied workflow of the template (a swap is a no-op for
  workflows that never mention the name: `comfyui-qwen-image/src/provision_models.py:35-43` comment).
- **Queue rule** = when the group is active, every `S[r]` is queued via the registry (minimax
  `:100-101`); a profile filename missing from the registry is a config error (exit 2).
- Multiple groups may exist (qwen: one per model family, so enabling one flag never downloads
  another family's swapped files; each group carries its own env, matching qwen's per-flag
  `precision_env`, `provision_models.py:89`).

### 5b. flags map

Walk mode entry (wan/minimax folder shape, or qwen explicit-list shape):

```jsonc
"download_wan21":        { "folders": ["Wan 2.1", "Infinite Talk"] },     // wan :24-30
"DOWNLOAD_QWEN_IMAGE":   { "workflows": ["Qwen_Workflow.json", ...],      // qwen workflows_registry.json
                           "default": true,
                           "extra_models": ["krea2_raw_bf16.safetensors"] }
```

Registry mode entry (copy lists only; model gating lives in the registry):

```jsonc
"download_ltx23":    { "copy": ["."],           "default": true  },
"download_ltx2_19b": { "copy": ["legacy_19b"],  "default": false },
"download_ltx25":    { "copy": ["LTX2.5"],      "default": false }
```

`default: true` follows `on_by_default` truthiness (only literal `"false"` disables, §3);
`default: false` (or absent) is opt-in.

### 5c. Worked example: minimax (walk mode, quant profiles)

Derived from `comfyui-minimax/src/start.sh:204-224`, `src/workflow_provisioner.py:22-51`,
`src/start.sh:143` (vae_approx), `src/start.sh:157-159` (nothing cloned at boot):

```json
{
  "template_repo": "https://github.com/Hearmeman24/comfyui-minimax.git",
  "branch": "master",
  "provisioning_mode": "walk",
  "flags": {
    "download_minimax_h3": { "folders": ["MiniMax H3"] }
  },
  "swap_groups": [
    {
      "env": "minimax_quant",
      "default": "int8",
      "flags": ["download_minimax_h3"],
      "profiles": {
        "int8": {
          "fl2va": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
          "ref2va": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
          "text_encoder": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        },
        "fp8": {
          "fl2va": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
          "ref2va": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
          "text_encoder": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        },
        "nvfp4": {
          "fl2va": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
          "ref2va": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
          "text_encoder": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        }
      }
    }
  ],
  "extra_model_paths": ["vae_approx"],
  "custom_nodes": { "target": "image", "repos": [] },
  "sage": true
}
```

(`nvfp4` resolves to the same files as `fp8` so pods already setting it keep booting,
`comfyui-minimax/src/workflow_provisioner.py:45-49`. The `--quant` CLI arg with argparse `choices`
dies; unknown env values warn and fall back to `int8`.)

### 5d. Worked example: ltx2 (registry mode)

Derived from `comfyui-ltx2/src/start.sh:195-260,378-417`, `src/build_manifest.py`,
`src/start.sh:119` (model_patches), `src/start.sh:92-101` (Director node boot pull):

```json
{
  "template_repo": "https://github.com/Hearmeman24/comfyui-ltx2.git",
  "branch": "main",
  "provisioning_mode": "registry",
  "flags": {
    "download_ltx23":    { "copy": ["."],          "default": true  },
    "download_ltx2_19b": { "copy": ["legacy_19b"], "default": false },
    "download_ltx25":    { "copy": ["LTX2.5"],     "default": false }
  },
  "variant_env": "lightweight_fp8",
  "swap_groups": [
    {
      "env": "lightweight_fp8",
      "default": "false",
      "flags": ["download_ltx2_19b"],
      "profiles": {
        "false": { "dit_19b": "ltx-2-19b-dev.safetensors" },
        "true":  { "dit_19b": "ltx-2-19b-dev-fp8.safetensors" }
      }
    }
  ],
  "extra_model_paths": ["model_patches"],
  "custom_nodes": {
    "target": "image",
    "repos": ["https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git"]
  },
  "sage": true
}
```

Notes: the swap group replaces the 19b sed rewrite (`comfyui-ltx2/src/start.sh:408-417`); profile
keys are the raw env values `"false"`/`"true"` because `lightweight_fp8` is a boolean-shaped env.
The 19b registry entries additionally carry `variant` full/fp8 so the DOWNLOAD side follows the same
switch (`build_manifest.py:77-81`); a group's queue rule and the registry variant gate select the
same file, by construction. `"sage": true` here is the plan's Q4 decision (sage turns ON for ltx2 at
migration; it is absent from ltx2 today). ltx2's remaining boot specials become hooks (§7): the
HF_TOKEN sanity check + LTX-2.5 gated preflight (`start.sh:190-239`) in `pre_download.sh`; the
`kornia==0.8.2` pin (`start.sh:461-469`) in `pre_launch.sh`. The Director node entry reproduces
ltx2's boot-time best-effort pull of the image-baked pack (`comfyui-ltx2/src/start.sh:92-101`; clone
URL confirmed at `comfyui-ltx2/Dockerfile:95`): the dir exists, so the loop's pull branch runs.

### 5e. src/runtime_nodes.json — packs every template gets

A JSON array in THIS repo, same entry syntax as `custom_nodes.repos`
(`"<url>"`, `"<url>|<sha>"`, `"<url>|force"`):

```json
[
  "https://github.com/Hearmeman24/ComfyUI-HearmemanAI-Upscale.git"
]
```

`start.sh` reads it before `template.json`'s list and clones both through the one loop. A pack
belongs here when every template should have it: one push plus a `stable` promotion reaches all of
them, instead of an identical one-line PR per template repo. A pack only one template needs stays
in that template's `custom_nodes.repos`.

Rules, all enforced by `tools/test_custom_node_list.py`:

- **Dedup is by directory name**, the same `basename <url> .git` the loop uses. A name on both
  lists is cloned once.
- **On a collision the TEMPLATE's entry wins**, at the runtime entry's position. The template is
  the more specific source and may carry a pin the runtime list does not.
- **A missing or malformed file costs the runtime packs, never the boot.** It is read on every pod
  of every template, so it fails open: unreadable → empty list, one warning on stderr, the
  template's own list still clones.
- **Keep the bar high.** These packs clone on every template, so favour ones with no dependencies:
  a `requirements.txt` here is a pip install on every boot of every pod, and one bad entry breaks
  four templates at once rather than one. Anything heavy or version-sensitive belongs in a
  template's own list, or in the base image.

---

## 6. pins.json

Per-template, repo root. Exactly two keys (plan D2):

```json
{
  "runtime_ref": "4f0c9a1e...40-hex-sha",
  "base_image": "hearmeman/comfyui-base:cu130-comfy0.32.0-torch2.11.0"
}
```

- `runtime_ref`: the `comfyui-runtime` commit the template's boot pins. Read by the baked
  `start_script.sh` at every boot (`git fetch origin $RUNTIME_REF && git reset --hard FETCH_HEAD`,
  plan §3). A branch name (`main`) is legal for instant family-wide deploy: supported, not default
  (plan Q2). Deploys on the next pod RESTART, no rebuild.
- `base_image`: the base tag the template's Dockerfile builds `FROM`. Consumed by CI as a build arg
  at `vN` tag time. Reaches only NEW pods through a tag + template repoint.
- `start.sh` prints both values at boot so every support log names them (plan D2).

Base tag grammar: `hearmeman/comfyui-base:<cuNNN>-comfy<ref>-torch<ver>` (plan §5c "Tagging"), e.g.
`cu128-comfy0.32.0-torch2.11.0`.

---

## 7. Hook contract: src/hooks/pre_download.sh, src/hooks/pre_launch.sh

Optional, per-template, at `$TEMPLATE_DIR/src/hooks/`. If present, the shared `start.sh` SOURCES
them (bash `source`, not exec; plan §2: "sourced if present"). Sourcing is the point: a hook may
`export` env vars that the provisioner then reads (ltx2's 2.5 preflight forces
`export download_ltx25=false` on a failed token probe, `comfyui-ltx2/src/start.sh:224,231`).

When they run (boot order in §9):

- `pre_download.sh`: immediately BEFORE the provisioner call. Everything earlier has happened:
  volume symlinks, `extra_model_paths.yaml`, `COMFYUI_VERSION` handling, the sage phase, the
  custom-node clone loop. Use it for download preflights, flag flips and template-specific installs
  (donors: ltx2's HF_TOKEN check + LTX-2.5 preflight `start.sh:190-239`; wan's conditional
  OpenRouter clone `start.sh:181-196`; qwen's Boogu-Image install + flash-attn fetch
  `start.sh:147-167`).
- `pre_launch.sh`: after the model phase, the CivitAI ID downloads and the node-requirements waits,
  immediately BEFORE the ComfyUI launch. Use it for last-mile pip pins and file fixups (donor:
  ltx2's `kornia==0.8.2` pin, `start.sh:461-469`).

Environment a hook may rely on (all exported by `start.sh` before sourcing):

| var | value |
|---|---|
| `TEMPLATE_DIR` | absolute path of the template repo clone (e.g. `/comfyui-minimax`) |
| `RUNTIME_DIR` | `/comfyui-runtime` |
| `NETWORK_VOLUME` | `/workspace`, or `/` when no volume |
| `PERSIST_ROOT` | `$NETWORK_VOLUME/ComfyUI` |
| `COMFYUI_DIR` | `/ComfyUI` |
| `WORKFLOW_DIR` | `$PERSIST_ROOT/user/default/workflows` |
| `CUSTOM_NODES_DIR` | the clone target dir per `template.json` |
| `HF_QUEUE_FILE` | manifest path (pre_download only) |
| PATH | `/opt/venv/bin` first; `pip`/`python3` are the venv's |
| `PIP_CONSTRAINT` | `/torch-constraint.txt`, set by the base image (donor `comfyui-qwen-image/Dockerfile:47-48`) |

Plus every RunPod template env var (flags, tokens, etc.).

A hook MUST NOT:

- call `exit` (it is sourced; `exit` kills the boot; use `return`),
- launch, restart or wait on ComfyUI,
- touch the `/ComfyUI` checkout's git state (`COMFYUI_VERSION` owns it, plan §5b),
- install/uninstall/move torch, torchvision, torchaudio, or unset `PIP_CONSTRAINT`,
- write into `$RUNTIME_DIR` or change `$PERSIST_ROOT` layout (frozen, spec §5),
- block unbounded: any network probe uses `curl --max-time` or equivalent,
- assume the working directory,
- treat its own failure as fatal: `start.sh` runs without `set -e` and does not gate the boot on a
  hook's success; a hook handles its own errors and prints its own one-line diagnostics.

Hooks run on EVERY boot, restarts included, and must be idempotent.

---

## 8. src/sage_probe.py (slice C)

Extracted from the duplicated heredoc probe (`comfyui-wan/src/start.sh:26-34`, byte-identical in
minimax `:36-44`): a REAL kernel launch, because import success is not evidence (the current wheel
imports fine on an H100 and then cannot launch, spec §1).

Invocation, no arguments:

```
python3 /comfyui-runtime/src/sage_probe.py
```

Behavior and exit codes:

| exit | case | printed line (exactly one, to stdout; D9 wording) |
|---|---|---|
| 0 | kernel launch succeeded | `SageAttention probe passed on sm<XX>` |
| 1 | arch is dispatch-supported but the probe raised (import error, launch failure, no CUDA device on a supported card). After this work this is a BUG, not a routine. | `SageAttention probe failed on sm<XX>, launching without it, report this` (when the capability itself is unreadable, print `sm??`) |
| 2 | unsupported arch: `torch.cuda.get_device_capability()` not in the dispatch set | `unsupported GPU arch sm<XX>, SageAttention off` |

The dispatch set is `{(8,0), (8,6), (8,9), (9,0), (12,0), (12,1)}`: exactly the arms upstream
`core.py:143-157` dispatches (sm80/86/89/90/120/121, verified at HEAD during spec drafting, spec §2).
sm86 is in the set although it needs no cubin (triton JIT path, plan D7). sm70 (V100), sm100/sm103
(B200/B300) exit 2: upstream has no arm for them and no wheel can fix it (plan §5).

The probe body is the frozen heredoc: fp16 `q = torch.randn(1, 8, 128, 64, device="cuda")`,
`sageattn(q, q.clone(), q.clone())`, `torch.cuda.synchronize()` (`comfyui-wan/src/start.sh:27-33`).
The arch check runs BEFORE importing sageattention, so exit 2 is reachable even when the wheel is
absent or broken.

Consumers:

- `start.sh` (§9 step 6): exit 0 adds `--use-sage-attention`; exit 1 or 2 launches without it. The
  probe prints the message; `start.sh` relays it at the pre-launch join and prints nothing extra.
  No retry, no source build (plan D9: the build subshell, the wait loop, `/tmp/sage_build_done` and
  the source fallback are all deleted). The only files are the two verdict files the install+probe
  subshell hands the join (E10).
- `tools/build_sage_wheel.sh` (slice E): runs the probe on the build pod after installing the fresh
  wheel; anything but exit 0 fails the build (plan §5 step 3b).
- The 0c throwaway-pod check and the paid probe matrix (plan Q5) require exit 0.

---

## 9. src/start.sh: boot order and every helper invocation (slice C)

Shape is minimax's `src/start.sh` (the most current, plan §2), minus every sage-build remnant.
Runtime scripts run IN PLACE from `/comfyui-runtime`; nothing is copied to `/` any more (§12).

Entry (from each template's baked `start_script.sh`, plan §3 step 4):

```
exec bash /comfyui-runtime/src/start.sh /comfyui-<template>
```

`$1` is `TEMPLATE_DIR`, the absolute path of the template repo clone. Everything else arrives via
env. `start.sh` reads `$TEMPLATE_DIR/template.json`, `$TEMPLATE_DIR/pins.json` (print both pins,
plan D2) and passes `$TEMPLATE_DIR/src/models_registry.json` to the provisioner.

Boot order (donor citations against minimax/wan; architecture.md §3):

1. tcmalloc preload + `/workspace/additional_params.sh` if present (minimax `start.sh:3-14`).
2. DNS preflight naming RunPod Global Networking in the error (spec §7 slice C; `CLAUDE.md` §6).
3. `NETWORK_VOLUME` resolve, JupyterLab launch (minimax `:19-24,78-84`).
4. Volume symlinks (`user`/`output`/`input`, plus `models` when `models_symlink` is true) and
   generated `extra_model_paths.yaml` from the base list + `extra_model_paths` additions, with a
   matching `mkdir -p` per category (minimax `:95-149`; qwen models symlink `start.sh:58`).
5. `COMFYUI_VERSION` handling: resolve target (`approved` reads `/comfyui-approved-ref`; `latest`
   calls the releases API; explicit ref as-is), compare `git rev-parse HEAD`, move only on mismatch,
   never move on resolve failure (plan §5b).
6. Sage phase, only when `template.json` `"sage": true` (plan D9; EXECUTION.md E10): read
   `torch.version.cuda` major synchronously, then background ONE subshell doing
   `pip install --no-deps --force-reinstall /opt/sage/cu<128|130>/sageattention-*.whl` ->
   `python3 /comfyui-runtime/src/sage_probe.py`, writing the probe's exit code and message to
   `/tmp/sage_verdict.rc` / `/tmp/sage_verdict.msg`. The subshell overlaps steps 7 through 14 and
   is joined in step 15, where exit 0 sets `SAGE_FLAG="--use-sage-attention"`.
7. CivitAI downloader: baked at `/usr/local/bin/download_with_aria.py` by the base image (donor
   `comfyui-qwen-image/Dockerfile` bake; plan §5c); `start.sh` keeps the git clone ONLY as an
   if-missing fallback for images built before the base exists (today's every-boot clone:
   wan `start.sh:144-148`).
8. Custom-node clone loop from `template.json` (`custom_nodes.repos`, syntax §5), then their
   requirements installs (backgrounded, PIDs collected and waited before launch, wan `:199-217,
   413-429`).
9. `source $TEMPLATE_DIR/src/hooks/pre_download.sh` if present (§7).
10. Provisioner:

    ```
    python3 /comfyui-runtime/src/provisioner.py \
        --template  "$TEMPLATE_DIR/template.json" \
        --registry  "$TEMPLATE_DIR/src/models_registry.json" \
        --workflows-src "$TEMPLATE_DIR/workflows" \
        --workflows-dst "$WORKFLOW_DIR" \
        --models-root   "$PERSIST_ROOT/models" \
        --manifest      "$HF_QUEUE_FILE"
    ```

    with `HF_QUEUE_FILE=/tmp/hf_download_queue.tsv`. Exit 2 or 1 prints one loud line; boot
    continues (missing models surface as notices, never as a dead pod).
11. Downloader, EXIT CODE CHECKED (wan does not check it today, `start.sh:257`; `CLAUDE.md` §3):

    ```
    python3 /comfyui-runtime/src/hf_download_manager.py "$HF_QUEUE_FILE"
    ```

    Nonzero: print one line naming the count of failures; boot continues.
12. CivitAI ID downloads (`CHECKPOINT_IDS_TO_DOWNLOAD` / `LORAS_IDS_TO_DOWNLOAD` env, comma-split,
    `replace_with_ids` placeholder skipped): `(cd "$TARGET_DIR" && download_with_aria.py -m "$MODEL_ID") &`
    then the aria2c wait loop (minimax `:227-263`); zip-to-safetensors rename after (minimax
    `:303-307`, with `LORAS_DIR` actually defined, which neither donor does).
13. Wait on backgrounded pip installs; onnxruntime CUDA-provider boot re-check and reinstall if
    clobbered (wan `:431-440`; stays per plan §5c: it guards boot-time installs).
14. `source $TEMPLATE_DIR/src/hooks/pre_launch.sh` if present (§7).
15. Sage join: `wait` on the step 6 subshell, read the verdict files, set `SAGE_FLAG` and record
    the `sage` / `sage_msg` report keys (a missing verdict fails safe to `probe_failed`). The
    launch line interpolates `SAGE_FLAG`, so the join MUST precede it. Then launch, once,
    `nohup`ed, never restarted to add a flag (`CLAUDE.md` §6):

    ```
    nohup python3 "$COMFYUI_DIR/main.py" --listen --enable-cors-header '*' \
        $SAGE_FLAG $EXTRA_PATHS_FLAG \
        > "$NETWORK_VOLUME/comfyui_${RUNPOD_POD_ID}_nohup.log" 2>&1 &
    ```

    (minimax `:330-332`), then the curl liveness loop, the numbered triage block, `sleep infinity`
    (minimax `:334-370`).

`PERSIST_ROOT` layout, env var names and flag semantics are FROZEN (spec §5): existing customer
volumes mount unmodified.

---

## 10. tools/build_sage_wheel.sh and tools/validate_models.py (invocation only)

Not called by `start.sh`; listed so slices E and D freeze the same surface.

- `build_sage_wheel.sh <cu128|cu130>`: run ON the build pod via the runpod-ssh MCP. Installs
  `-r torch/cu<NNN>.txt` and asserts the venv equals the canonical trio; exports
  `SAGE_COMMIT=d1a57a546`, `TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 12.0 12.1"`, `EXT_PARALLEL=4`,
  `NVCC_APPEND_FLAGS="--threads 8"`, `MAX_JOBS=32` (compile invocation lifted from
  `comfyui-minimax/src/sage_build.sh:22,28,81-87`; the cache-key logic is dropped, plan §5);
  `pip wheel --no-deps --no-build-isolation`; then gates on-pod: cuobjdump cubin assertions
  (`_qattn_sm80` has sm_80; `_qattn_sm89` has sm_89 + sm_120 + sm_121; `_qattn_sm90` has sm_90a) and
  a `sage_probe.py` exit 0 (plan §5 step 3).
- `validate_models.py` (slice D, reconcile onto ltx2's 191-line copy): CI-side gate, run with a real
  torch venv. Must walk `definitions.subgraphs[].nodes`, compare the FULL widget value including any
  path prefix, check `gated` entries against the HF model API file list (not a HEAD), and size-check
  presigned URLs with a ranged GET (`Range: bytes=0-0`, expect 206) (`CLAUDE.md` §3, §9; spec §7
  slice D). It rejects `dest_subdir` (§4).

---

## 11. Environment variables the runtime owns

| var | default | meaning |
|---|---|---|
| `COMFYUI_VERSION` | `approved` | plan §5b; `approved` actively restores to `/comfyui-approved-ref` |
| `HF_TOKEN` | unset | user-supplied only; gates `gated` registry entries; never baked |
| `CHECKPOINT_IDS_TO_DOWNLOAD`, `LORAS_IDS_TO_DOWNLOAD` | `replace_with_ids` | CivitAI ID lists (wan `start.sh:260-263`) |
| `civitai_token` / `CIVITAI_TOKEN` / `CIVITAI_API_KEY` | unset | all three accepted (`CLAUDE.md` §3) |
| `HF_LOCAL_STAGE` | `/hf_stage` | where models download to and live until `volume_sync.py` copies them to the volume |
| `COMFY_EXTRA_ARGS` | unset | appended verbatim to the ComfyUI launch, word-split. Escape hatch for upstream bugs without cutting a tag; echoed at boot when set (`src/start.sh:628-637`) |
| per-template flags, quant/precision/variant envs | per `template.json` | §3, §5 |
| `CUDA_VARIANT` | dead at boot | the sage path no longer reads it (selection is by `torch.version.cuda`, plan D9); the env survives only as a Docker build ARG in templates |

---

## 12. Deliberate departures from today's code (plan wins over code)

Slice agents: these are DESIGN CHANGES, not oversights. Do not restore the old behavior.

1. **Runtime scripts run from `/comfyui-runtime`, not copied to `/`.** Today every entrypoint
   `cp -f`s scripts to `/` and start.sh calls `/workflow_provisioner.py` etc.
   (`comfyui-wan/src/start.sh:248-257`). New invocation paths are §9. (plan §3 step 4)
2. **The boot-time sage source build is deleted, not demoted**: no `sage_build.sh` port, no
   `.sage_wheel_cache`, no background subshell (`wan start.sh:50-63`), no
   `/tmp/sage_build_done` wait loop (`wan start.sh:450-463`, `minimax start.sh:311-324`). The
   compile logic survives once, as `tools/build_sage_wheel.sh`. (plan D9, §5)
3. **The `CUDA_VARIANT = cu130` install gate dies** (`comfyui-wan/src/start.sh:37`): it is the bug
   that makes every default-image wan pod compile sage on every boot. Selection is by
   `torch.version.cuda` major. (plan D9)
4. **Non-HF manifest URLs are downloaded, not silently dropped** (`wan hf_download_manager.py:56-58`
   dies; §1 routing). The inline aria2c blocks in `start.sh` for the skin upscaler, RIFE and
   2xLiveActionV1_SPAN (`wan start.sh:317-359`, `ltx2 start.sh:256-259`) die: those files become
   registry entries pointing at the HF mirror (plan D3), with `upstream_url` recorded.
5. **Provisioner flags move from argv to env** (§3). The per-template flag-translation shell blocks
   die.
6. **The downloader's exit code is checked** in `start.sh` (wan does not, `start.sh:257`;
   `CLAUDE.md` §3).
7. **`min_size_mb` reaches the downloader** via the manifest's new third field (§1); the flat 10 MB
   floor becomes the default, not the rule.
8. **qwen's plain-text precision swap** (`provision_models.py:58-62`) is replaced by the JSON-aware
   widget + `properties.models` rewrite (minimax `:141-160`); spec §7 slice B makes both rewrites
   mandatory.
9. **minimax's `--quant` argparse arg** (with `choices` hard-failing on unknown values,
   `workflow_provisioner.py:64`) becomes the env-driven swap group; unknown values warn and use the
   default (§5a).
10. **`dest_subdir` is dead** (§4; plan D5). No shim.
11. **ltx2's `timeout 4000` downloader wrapper** (`start.sh:250`) is not ported; the watchdog it
    duplicates is in the downloader (§2).
12. **Manager `config.ini` seeding is not ported** (wan still writes it, `start.sh:377-402`; minimax
    already dropped it as all-defaults, `minimax start.sh:279-283`; minimax's shape wins).
13. **ComfyUI is never mutated ad hoc at boot**: qwen's unconditional `git checkout master && git
    pull` (`comfyui-qwen-image/src/start.sh:92-95`) is replaced by `COMFYUI_VERSION` (plan §5b).
14. **The every-boot CivitAI downloader clone** (`wan start.sh:144-148`) becomes a base-image bake
    with an if-missing boot fallback (plan §5c).

---

## 13. Open DECISION NEEDED index (scan targets)

1. §1: manifest third column `min_size_mb`.
2. §3: registry-mode workflow copies preserve relative paths (ltx2 previously flattened).
3. §3: LTX-2.5 workflows no longer copied when the preflight forces the flag off.
4. `torch/cu128.txt`: pin the cu128 trio to torch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0
   (newest coherent stable on the cu128 index as of 2026-08-12, version-identical to minimax's
   validated cu130 pin). The cu128 templates are unpinned today, so this is a new pin, not a copy.
