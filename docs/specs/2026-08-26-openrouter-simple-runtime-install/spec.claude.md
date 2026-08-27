# Install OpenRouter Simple on every shared-runtime template

- **Work type:** `feature/app`
- **Status:** `implemented locally`; release remains gated
- **Review surface:** [`spec.human.md`](./spec.human.md)

## 1. Problem / Context

OpenRouter Simple is installed manually on the current H200 pod, but the user wants it owned by `comfyui-runtime` so it is present on every shared-runtime ComfyUI template rather than relying on an in-pod copy.

## 2. Approach & Why

- `src/runtime_nodes.json` is the authoritative list of packs supplied to every shared-runtime template (`CONTRACTS.md:525-539`).
- The boot script reads that list before each template's own `custom_nodes.repos`, merges both by checkout directory name, and gives a colliding template entry precedence (`src/start.sh:484-550`).
- Each resolved entry is cloned when missing or pulled when present, and requirements are installed only when its Git HEAD changes (`src/start.sh:552-601`).
- OpenRouter Simple is absent from every active template's own runtime clone list: wan, minimax, and qwen-image list no template-specific runtime repos, while ltx2 lists four different packs (`../comfyui-wan/template.json:72-75`, `../comfyui-minimax/template.json:44-47`, `../comfyui-qwen-image/template.json:82-85`, `../comfyui-ltx2/template.json:24-32`). A search of all four sibling Dockerfiles and `src/` trees also found no baked declaration on 2026-08-26.
- The node declares only `aiohttp`, `numpy`, and Pillow version ranges (`../ComfyUI-OpenRouter-Simple/requirements.txt:1-3`); the runtime already backgrounds changed-checkout requirements installs and waits for them before ComfyUI launch (`src/start.sh:590-600,712-720`).

## 3. Acceptance Criteria

- [x] OpenRouter Simple is declared in the runtime-owned list consumed by every shared-runtime template. → (ask: "Make sure this node is always installed in the comfyui-runtime repo")
- [x] The shipped-list regression fails if the OpenRouter checkout is absent or duplicated. → (ask: "Make sure")
- [x] The change remains local until an explicitly authorized GitHub push and later `stable` promotion. → (ask: "in the comfyui-runtime repo")

## 4. Scope & Non-Goals

**In scope:** `src/runtime_nodes.json:1-4`, the shipped-list regression in `tools/test_custom_node_list.py:215-227`, and this historical spec.

**Non-goals:** no template-repo edits, no node source vendoring, no base-image rebuild, no OpenRouter key handling, no workflow execution, no automatic `stable` promotion, and no release of the local OpenRouter Simple v0.2.0 commit.

## 5. Key Decisions & Constraints

- **Decided:** use an unpinned public Git URL, matching the existing runtime-owned entries (`src/runtime_nodes.json:1-4`).
- **Constraint / must-not-break:** directory names must remain unique because list merging and clone targets are keyed by `basename <url> .git` (`src/start.sh:526-557`).
- **Constraint / must-not-break:** node dependencies stay in the node repository and use the existing constrained, changed-checkout install path (`src/start.sh:567-600`).
- **Mirror existing:** the `ComfyUI-LoRABlockSurgeon` shared entry in `src/runtime_nodes.json:3`.
- **Scale:** every new pod boot across four active public templates; the relevant bottlenecks are one shallow Git clone and one constrained requirements install on a fresh checkout.

## 6. Code Surface Map

- `src/runtime_nodes.json:1-4` — runtime-owned pack registry.
- `src/start.sh:484-603` — authoritative merge, clone/update, and dependency-install loop; unchanged.
- `tools/test_custom_node_list.py:215-227` — shipped-list validation seam.
- `CONTRACTS.md:525-553` — frozen runtime-node ownership and dependency policy; unchanged.

## 7. Ultracode Dispatch Notes

**Build first (sequential — freezes interfaces before any parallelism):**
- Keep the existing URL-string schema and checkout-name deduplication contract unchanged.

**Parallel slices:**
- None. The registry entry and its assertion are one small coherent slice.

**⛓ Collision audit:** One slice owns the list and its focused test; no shared mutable surface is split.

**Each agent must:** add the red-capable assertion, add the entry, and run focused plus full verification.

```yaml
dispatch:
  frozen: ["src/start.sh", "CONTRACTS.md", "ARCHITECTURE.md", "../comfyui-wan", "../comfyui-minimax", "../comfyui-qwen-image", "../comfyui-ltx2"]
  slices:
    - {key: runtime_node_entry, writes: ["src/runtime_nodes.json", "tools/test_custom_node_list.py"]}
  testRunner: "python3 tools/test_custom_node_list.py"
```

## 8. Assumptions & Open Questions

None. The ownership mechanism, dependency path, active-template scope, and lack of sibling declarations were verified from the current checkouts.
