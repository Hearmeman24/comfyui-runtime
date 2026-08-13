#!/usr/bin/env python3
"""Derive the extra_model_paths category list from the pinned ComfyUI tree.

start.sh used to hand-manage a frozen 14-category list; it drifted every time
ComfyUI grew a category (14 in wan's day, 27 keys in v0.32.0). This helper
asks the ComfyUI tree that will actually run: it imports folder_paths from
the given directory and emits one "key<TAB>relpath" line per registered dir
under models_dir. Multi-dir keys emit one line per dir, which is how the
legacy alternates ride along under their REAL key:

    text_encoders     -> text_encoders, clip
    diffusion_models  -> unet, diffusion_models
    controlnet        -> controlnet, t2i_adapter

clip and unet are in folder_paths.map_legacy, t2i_adapter is NOT: none of the
three may ever appear as a yaml key of its own, only as an extra dir under
its canonical key. Dirs hanging off base_path instead of models_dir
(custom_nodes, datasets) are excluded; start.sh handles custom_nodes with its
bespoke base_path-relative line.

Usage:
    model_paths.py <comfyui_dir>    derive from the tree (exit 0)
    model_paths.py --fallback       print the frozen v0.32.0 superset (exit 0)

On ANY failure (bad args, missing/broken tree, unexpected layout) the frozen
fallback is printed and the exit code is 3: start.sh report_warns on 3 but
always gets a full list, never an empty or partial one (boot failures warn
and continue, decision #19).

Stdlib only; importable and runnable without a pod (tools/test_model_paths.py).
"""
import os
import sys

# The full derivation against ComfyUI v0.32.0 (folder_paths.py:25-67), frozen
# 2026-08-13: 25 models_dir keys, 28 dirs. test_model_paths.py checks this
# against a live derivation when COMFYUI_TREE points at a v0.32.0 tree, and
# skips loudly when it does not (CI has no tree). Treat the list as frozen to
# v0.32.0: it is the fallback, not a moving mirror of whatever ComfyUI is
# checked out. Rebuild it deliberately when the base image's COMFYUI_REF moves.
FALLBACK = (
    ("checkpoints", "checkpoints"),
    ("configs", "configs"),
    ("loras", "loras"),
    ("vae", "vae"),
    ("text_encoders", "text_encoders"),
    ("text_encoders", "clip"),
    ("diffusion_models", "unet"),
    ("diffusion_models", "diffusion_models"),
    ("clip_vision", "clip_vision"),
    ("style_models", "style_models"),
    ("embeddings", "embeddings"),
    ("diffusers", "diffusers"),
    ("vae_approx", "vae_approx"),
    ("controlnet", "controlnet"),
    ("controlnet", "t2i_adapter"),
    ("gligen", "gligen"),
    ("upscale_models", "upscale_models"),
    ("latent_upscale_models", "latent_upscale_models"),
    ("hypernetworks", "hypernetworks"),
    ("photomaker", "photomaker"),
    ("classifiers", "classifiers"),
    ("model_patches", "model_patches"),
    ("audio_encoders", "audio_encoders"),
    ("background_removal", "background_removal"),
    ("frame_interpolation", "frame_interpolation"),
    ("geometry_estimation", "geometry_estimation"),
    ("optical_flow", "optical_flow"),
    ("detection", "detection"),
)


def derive(comfy_dir):
    """Import folder_paths from comfy_dir and return [(key, relpath)] for
    every registered dir under models_dir, in registration order. Raises on
    anything unexpected; the caller decides what a failure means."""
    comfy_dir = os.path.abspath(comfy_dir)
    if not os.path.isfile(os.path.join(comfy_dir, "folder_paths.py")):
        raise FileNotFoundError("no folder_paths.py in %s" % comfy_dir)
    # comfy.cli_args parses argv at import when args_parsing is on; a bare
    # argv keeps the import inert either way (spec D5, probed on v0.32.0).
    sys.argv = [sys.argv[0] if sys.argv else "model_paths"]
    sys.path.insert(0, comfy_dir)
    import folder_paths
    imported_from = os.path.abspath(getattr(folder_paths, "__file__", ""))
    if not imported_from.startswith(comfy_dir + os.sep):
        raise ImportError("folder_paths resolved to %s, not the requested tree"
                          % imported_from)
    models_dir = os.path.abspath(folder_paths.models_dir)
    pairs = []
    for key, entry in folder_paths.folder_names_and_paths.items():
        for d in entry[0]:
            rel = os.path.relpath(os.path.abspath(d), models_dir)
            if rel == "." or rel.startswith(".." + os.sep) or rel == "..":
                continue  # hangs off base_path (custom_nodes, datasets)
            pairs.append((key, rel))
    if not pairs or "checkpoints" not in {k for k, _ in pairs}:
        raise RuntimeError("derived list from %s looks wrong: %r"
                           % (comfy_dir, pairs))
    return pairs


def emit(pairs):
    for key, rel in pairs:
        print("%s\t%s" % (key, rel))


def main(argv):
    if len(argv) == 2 and argv[1] == "--fallback":
        emit(FALLBACK)
        return 0
    if len(argv) != 2:
        print("model_paths: usage: model_paths.py <comfyui_dir> | --fallback; "
              "printing the frozen fallback list", file=sys.stderr)
        emit(FALLBACK)
        return 3
    try:
        pairs = derive(argv[1])
    except Exception as exc:
        print("model_paths: derivation from %s failed (%r); printing the "
              "frozen v0.32.0 fallback list" % (argv[1], exc), file=sys.stderr)
        emit(FALLBACK)
        return 3
    emit(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
