#!/usr/bin/env python3
"""Boot deployment report renderer (EXECUTION.md items N1 + N2).

Boot failures warn and continue (decision #19), so nothing ever stops for the
customer to notice. This report, printed once at the very end of boot, is the
only place they learn that something degraded. It replaces the old single
"ComfyUI is Ready" line.

The report body is generated ONCE and rendered twice from the same string:
to stdout (the pod log), and into the troubleshooting note, one of three
single-MarkdownNote "workflows" (welcome, adding models, troubleshooting)
written to the customer's workflow list, so the current boot's report is one
click away inside ComfyUI itself. All three are re-rendered on every boot.

Format: compact always, expand on failure. A clean boot is nine one-line
rows. Only rows with something wrong expand, and every expansion names the
CONSEQUENCE, not just the fact.

Inputs (all optional; missing inputs degrade to "unknown" rows, they never
crash the boot):
  --state             TSV written by start.sh helpers: lines are
                        set\t<key>\t<value> | warn\t<text> | off\t<name>\t<how>
  --template          the template's template.json (workflow set labels)
  --manifest          the download manifest the provisioner wrote (section 1
                      format); the renderer audits each dest against its floor
  --provision-status  JSON written by provisioner.py when
                      PROVISION_STATUS_FILE is set
  --hf-status         JSON written by hf_download_manager.py when
                      HF_STATUS_FILE is set (failure reasons)
  --skeleton-dir      directory holding the three note skeletons
                      (src/note_welcome.md, src/note_civitai.md,
                      src/note_troubleshooting.md)
  --note-sections     per-template markdown inserted into the welcome note
                      (missing file is fine: no template ships one until its
                      migration commit)
  --notes-root        the customer's workflow directory; each note is written
                      as <notes-root>/<folder>/<title>.json

Customer-facing style: plain, short sentences, no emoji, no em or en dashes.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

RULE = "=" * 60
LABEL_W = 11          # "  " + label.ljust(11) puts every value at column 13
CONT = " " * 13       # continuation lines align under the values
DEFAULT_MIN_SIZE_MB = 10.0

MODEL_CONSEQUENCE = "Workflows using this model will error."
SAGE_CONSEQUENCE = "Workflows still run; generation is slower without it."

# The three customer notes, in the order they must appear in the workflow
# list. Notes live in folders because the workflow tree lists folders before
# leaf workflows, so a bare file at the root would sort below every workflow
# set folder. "!" sorts before digits and letters, floating the notes above
# set folders like "Wan 2.2 I2V"; the digit fixes the order among the three
# without leaning on how runs of "!" compare against each other.
# (skeleton file, folder name, note title)
NOTES = [
    ("note_welcome.md", "!1 Welcome", "Welcome"),
    ("note_civitai.md", "!2 Adding Models", "Adding Models"),
    ("note_troubleshooting.md", "!3 Troubleshooting", "Troubleshooting"),
]

# Arch families upstream SageAttention has no dispatch arm for
# (CONTRACTS.md section 8; fact C2: the sm100 compile work was reverted).
UNSUPPORTED_GPUS = {"100": "B200/B300", "103": "B200/B300", "70": "V100"}


def row(label: str, value: str) -> str:
    return f"  {label.ljust(LABEL_W)}{value}"


def read_state(path):
    """Parse the start.sh state TSV into (kv, warnings, offs)."""
    kv, warnings, offs = {}, [], []
    if not path:
        return kv, warnings, offs
    try:
        text = Path(path).read_text()
    except OSError:
        return kv, warnings, offs
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and fields[0] == "set":
            kv[fields[1]] = fields[2]
        elif len(fields) >= 2 and fields[0] == "warn":
            warnings.append(fields[1])
        elif len(fields) >= 3 and fields[0] == "off":
            offs.append((fields[1], fields[2]))
    return kv, warnings, offs


def read_json(path):
    if not path:
        return None
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_manifest(path):
    """Same line rules as the downloader (CONTRACTS.md section 1)."""
    entries = []
    if not path or not Path(path).is_file():
        return None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "\t" not in line:
            continue
        fields = line.split("\t")
        floor_mb = DEFAULT_MIN_SIZE_MB
        if len(fields) >= 3 and fields[2].strip():
            try:
                floor_mb = float(fields[2])
            except ValueError:
                pass
        entries.append({"url": fields[0], "dest": Path(fields[1]),
                        "floor": int(floor_mb * 1024 * 1024)})
    return entries


def sm_of(kv) -> str:
    m = re.search(r"sm(\d+)", kv.get("sage_msg", ""))
    return m.group(1) if m else ""


def short_reason(name: str, url: str, hf_status) -> str:
    entry = (hf_status or {}).get(name) or {}
    error = entry.get("error") or ""
    host = urlparse(entry.get("url") or url or "").netloc or "the source"
    if "404" in error:
        return f"404 from {host}"
    if "403" in error:
        return f"403 from {host} (access denied)"
    if "401" in error:
        return f"401 from {host} (needs a valid HF_TOKEN)"
    if "stalled" in error:
        return "stalled, no progress for 5 minutes"
    if "deadline" in error:
        return "download deadline exceeded"
    if error:
        return error[:60]
    return "not downloaded (see the boot log above)"


def comfy_row(kv) -> str:
    ver = kv.get("comfy_version", "").strip()
    sha = kv.get("comfy_sha", "").strip()
    mode = kv.get("comfy_mode", "").strip()
    what = f"v{ver}" if ver else (sha[:7] if sha else "unknown")
    return f"{what} ({mode})" if mode else what


def sage_rows(kv) -> list:
    state = kv.get("sage", "")
    sm = sm_of(kv)
    if state == "enabled":
        detail = f"sm{sm}, baked wheel" if sm else "baked wheel"
        return [row("SageAttn", f"enabled ({detail})")]
    if state == "off_template":
        return [row("SageAttn", "off (not used by this template)")]
    if state == "unsupported":
        who = UNSUPPORTED_GPUS.get(sm)
        tail = f"{who} unsupported" if who else "this GPU is unsupported"
        arch = f"sm{sm}" if sm else "this GPU arch"
        return [row("SageAttn", f"DISABLED ({arch} has no kernel; {tail})"),
                CONT + SAGE_CONSEQUENCE]
    if state == "probe_failed":
        where = f" on sm{sm}" if sm else ""
        return [row("SageAttn",
                    f"DISABLED (probe failed{where}; this is a bug, "
                    "please report it)"),
                CONT + SAGE_CONSEQUENCE]
    return [row("SageAttn", "unknown (sage phase did not run; see the boot log)")]


def model_rows(manifest, provision, hf_status) -> list:
    if manifest is None and provision is None:
        return [row("Models", "unknown (provisioner did not run; "
                              "see the boot log)"),
                CONT + "Bundled workflows may be missing their models."]
    queued = manifest or []
    skipped = len((provision or {}).get("skipped", []))
    total = len(queued) + skipped
    if total == 0:
        return [row("Models", "none requested (no download flags enabled)")]
    failed = [e for e in queued
              if not (e["dest"].is_file()
                      and e["dest"].stat().st_size >= e["floor"])]
    downloaded = total - len(failed)
    if not failed:
        return [row("Models", f"{total}/{total} downloaded")]
    lines = [row("Models",
                 f"{downloaded}/{total} downloaded, {len(failed)} FAILED")]
    for e in failed:
        name = e["dest"].name
        lines.append(f"     FAILED  {name}  "
                     f"{short_reason(name, e['url'], hf_status)}")
        lines.append(CONT + MODEL_CONSEQUENCE)
    return lines


def volume_copy_rows(manifest) -> list:
    """Models still living on local disk, waiting for the background copy.

    A staged model is a symlink into /hf_stage: fully usable right now, but it
    does not survive the pod. The report renders once, before the copy is done,
    so it must state what is pending rather than imply everything is durable.
    The completion line lands in the boot log, not here.
    """
    pending = []
    for e in manifest or []:
        dest = e["dest"]
        try:
            if dest.is_symlink() and dest.exists():
                pending.append(dest.stat().st_size)
        except OSError:
            continue
    if not pending:
        return []
    n = len(pending)
    total = sum(pending)
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            size = f"{total:.1f}{unit}"
            break
        total /= 1024
    else:
        size = f"{total:.1f}TB"
    return [row("Volume copy", f"{n} model{'s' if n != 1 else ''}, {size} "
                               f"still copying"),
            CONT + "They work now. Wait for the "
                   "\"safe to restart\" line in the log before restarting."]


def workflow_row(provision, template) -> str:
    if provision is None:
        return row("Workflows", "unknown (provisioner did not run)")
    enabled = provision.get("enabled_flags", [])
    if not enabled:
        return row("Workflows", "none (no workflow sets enabled)")
    flags_map = (template or {}).get("flags", {})
    labels = []
    for flag in enabled:
        cfg = flags_map.get(flag, {})
        folders = cfg.get("folders")
        if folders:
            labels.extend(folders)
        else:
            labels.append(re.sub(r"^download_", "", flag,
                                 flags=re.IGNORECASE).replace("_", " "))
    sets = len(enabled)
    return row("Workflows",
               f"{', '.join(labels)} ({sets} set{'s' if sets != 1 else ''})")


def warning_rows(warnings) -> list:
    if not warnings:
        return [row("Warnings", "none")]
    lines = [row("Warnings", str(len(warnings)))]
    for w in warnings:
        lines.append(f"     WARN    {w}")
    return lines


def header(kv) -> str:
    ready = kv.get("ready", "")
    if ready == "true":
        pod_id = kv.get("pod_id", "").strip()
        if pod_id:
            where = f"https://{pod_id}-8188.proxy.runpod.net"
        else:
            where = "on port 8188 (pod id unknown, no proxy URL)"
        return f"  ComfyUI is ready   {where}"
    if ready == "false":
        return "  ComfyUI FAILED to start   see the troubleshooting tips above"
    return "  ComfyUI status unknown (liveness check did not run)"


def render_report(kv, warnings, manifest, provision, hf_status,
                  template) -> str:
    gpu = kv.get("gpu_name", "").strip() or "unknown"
    sm = sm_of(kv)
    lines = [RULE, header(kv), RULE,
             row("ComfyUI", comfy_row(kv)),
             row("Runtime", "comfyui-runtime @ "
                 + (kv.get("runtime_sha", "").strip()[:7] or "unknown")),
             row("Base", kv.get("base_image", "unknown").split(":")[-1]),
             row("GPU", f"{gpu} (sm{sm})" if sm else gpu)]
    lines += sage_rows(kv)
    lines += model_rows(manifest, provision, hf_status)
    lines += volume_copy_rows(manifest)
    lines.append(workflow_row(provision, template))
    lines += warning_rows(warnings)
    lines.append(RULE)
    return "\n".join(lines)


def render_note(skeleton_path, sections_path, kv, offs, report) -> str:
    md = Path(skeleton_path).read_text()
    name = kv.get("template_name", "").strip()
    name = re.sub(r"^comfyui-", "", name) or "ComfyUI"
    if offs:
        disabled = "\n".join(f"- **{n}**: {how}" for n, how in offs)
    else:
        disabled = ("Nothing is switched off on this pod. "
                    "Every default feature of this template is active.")
    sections = ""
    if sections_path and Path(sections_path).is_file():
        sections = Path(sections_path).read_text().strip()
    md = md.replace("{{TEMPLATE_NAME}}", name)
    md = md.replace("{{DEPLOYMENT_REPORT}}", report)
    md = md.replace("{{DISABLED_FEATURES}}", disabled)
    md = md.replace("{{TEMPLATE_SECTIONS}}", sections)
    return re.sub(r"\n{3,}", "\n\n", md)


def note_workflow(md: str, title: str, seq: int) -> dict:
    """A one-node workflow in the UI format: a single MarkdownNote. The node
    type is registered by comfyui-frontend-package 1.48.7, the exact frontend
    ComfyUI v0.32.0 pins (requirements.txt at that tag); MarkdownNote is a
    virtual frontend node with its text in widgets_values[0], the same shape
    the tree's own blueprints ship. A note-only workflow has no API format."""
    return {
        "id": f"8e5a2f1c-0000-4000-8000-hearmemannot{seq}",
        "revision": 0,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [{
            "id": 1,
            "type": "MarkdownNote",
            "pos": [40, 40],
            "size": [620, 940],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "title": title,
            "properties": {},
            "widgets_values": [md],
            "color": "#432",
            "bgcolor": "#653",
        }],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--template")
    ap.add_argument("--manifest")
    ap.add_argument("--provision-status")
    ap.add_argument("--hf-status")
    ap.add_argument("--skeleton-dir")
    ap.add_argument("--note-sections")
    ap.add_argument("--notes-root")
    args = ap.parse_args(argv)

    kv, warnings, offs = read_state(args.state)
    report = render_report(
        kv, warnings,
        read_manifest(args.manifest),
        read_json(args.provision_status),
        read_json(args.hf_status),
        read_json(args.template),
    )
    print(report)

    if args.notes_root and args.skeleton_dir:
        for seq, (skeleton, folder, title) in enumerate(NOTES, 1):
            try:
                md = render_note(Path(args.skeleton_dir) / skeleton,
                                 args.note_sections, kv, offs, report)
                out = Path(args.notes_root) / folder / f"{title}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(note_workflow(md, title, seq),
                                          indent=2, ensure_ascii=False))
            except OSError as e:
                # The report already reached the log; a note failure must not
                # look like a boot failure.
                print(f"[boot-report] could not write the {title} note: {e}",
                      file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
