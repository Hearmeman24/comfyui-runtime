# Install OpenRouter Simple on every shared-runtime template

**Type:** `feature/app` · **Full spec:** [`spec.claude.md`](./spec.claude.md)

## ✅ What you'll see when this is done

Every wan, minimax, qwen-image, and ltx2 pod that boots from the promoted shared runtime clones or updates `ComfyUI-OpenRouter-Simple` before ComfyUI launches.

## 🪤 Gotchas

- This is a tier-1 runtime change. Merging it to `main` installs nothing by itself; moving `stable` makes it reach all four templates together on their next boot.
- The node has a small `requirements.txt`. The existing runtime installs it only after a fresh clone or changed checkout, waits before launch, and keeps the base image's pip constraint active.
- The public node repository currently serves v0.1.0 from `main`; the local v0.2.0 regenerate commit has not been pushed yet. The runtime entry tracks the public repository's `main`, as existing shared node entries do.

## Done when

- [x] OpenRouter Simple is declared in the runtime-owned node list used by every shared-runtime template.
- [x] A regression fails if the shipped runtime list drops or duplicates the OpenRouter Simple checkout name.
- [x] Runtime verification passes without changing any template-specific node list.
- [x] No `stable` promotion occurs without explicit authorization.

## The plan

1. Add a red assertion for the required runtime-owned checkout.
2. Register the public OpenRouter Simple repository in `src/runtime_nodes.json`.
3. Run the focused clone-loop suite and the full CI-equivalent runtime verification.
4. Commit the release-ready runtime change; hold GitHub push/PR and `stable` promotion for explicit authorization.
