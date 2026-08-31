#!/usr/bin/env python3
"""Self-check for the boot deployment report renderer (src/boot_report.py).

The report is the ONLY place a customer learns something degraded: boot
failures warn and continue (EXECUTION.md decision #19), so nothing else stops
for them to notice. These tests pin the two acceptance behaviors down:

  - a clean boot renders COMPACT: one line per row, nothing expanded;
  - anything degraded expands IN ITS ROW and always names the CONSEQUENCE,
    not just the fact ("Workflows using this model will error.").

Also covered: the three markdown-note "workflows" the renderer writes
(welcome, adding models, troubleshooting; one MarkdownNote node each, ComfyUI
UI format, re-rendered every boot with the live report injected into
troubleshooting), the no-flags boot (the notes must still be written), the
missing-pod-id fallback (no broken URL), and the customer-facing style rule
(no em or en dashes anywhere).

Everything runs against synthetic fixtures in a tempdir: no network, no GPU,
no pod. Run: python3 tools/test_boot_report.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RENDERER = REPO / "src" / "boot_report.py"
SKELETON_DIR = REPO / "src"

# Order and naming must match boot_report.NOTES: welcome, then the models
# guide, then troubleshooting, all sorting above the workflow set folders.
NOTE_FOLDERS = ["!1 Welcome", "!2 Adding Models", "!3 Troubleshooting"]
NOTE_TITLES = ["Welcome", "Adding Models", "Troubleshooting"]
PROVISIONER = REPO / "src" / "provisioner.py"

sys.path.insert(0, str(REPO / "src"))
import boot_report as br  # noqa: E402

MB = 1024 * 1024

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    assert cond, msg
    CHECKS += 1


# ---------------------------------------------------------------- fixtures

CLEAN_STATE = [
    ("set", "template_name", "comfyui-wan"),
    ("set", "pod_id", "abc123"),
    ("set", "ready", "true"),
    ("set", "comfy_version", "0.34.0"),
    ("set", "comfy_sha", "c2bcbecd82ec5ae66594340b395c24ef0217b238"),
    ("set", "comfy_mode", "approved"),
    ("set", "runtime_sha", "28a4ca1f00000000000000000000000000000000"),
    ("set", "base_image", "hearmeman/comfyui-base:cu128-comfy0.34.0-torch2.11.0"),
    ("set", "gpu_name", "NVIDIA H200"),
    ("set", "sage", "enabled"),
    ("set", "sage_msg", "SageAttention probe passed on sm90"),
]


def write_state(d: Path, rows) -> Path:
    p = d / "boot_report_state.tsv"
    p.write_text("".join("\t".join(r) + "\n" for r in rows))
    return p


def write_manifest(d: Path, entries) -> Path:
    """entries: list of (url, dest, min_size_mb or None)."""
    p = d / "hf_download_queue.tsv"
    lines = []
    for url, dest, floor in entries:
        line = f"{url}\t{dest}"
        if floor is not None:
            line += f"\t{floor}"
        lines.append(line)
    p.write_text("\n".join(lines) + ("\n" if lines else ""))
    return p


def write_json(d: Path, name: str, payload) -> Path:
    p = d / name
    p.write_text(json.dumps(payload, indent=1))
    return p


TEMPLATE = {
    "provisioning_mode": "walk",
    "flags": {
        "download_wan22": {"folders": ["Wan 2.2 I2V", "Wan Animate"]},
        "download_extra": {"workflows": ["Extra.json"]},
    },
}


def note_md(root: Path, which: str) -> str:
    """Read the rendered markdown of one note by its title."""
    folder = NOTE_FOLDERS[NOTE_TITLES.index(which)]
    doc = json.loads((root / folder / f"{which}.json").read_text())
    return doc["nodes"][0]["widgets_values"][0]


def render(d: Path, state_rows=None, manifest_entries=None,
           provision_status=None, hf_status=None, template=None,
           notes_root=None, sections=None):
    cmd = [sys.executable, str(RENDERER),
           "--state", str(write_state(d, state_rows or CLEAN_STATE))]
    if manifest_entries is not None:
        cmd += ["--manifest", str(write_manifest(d, manifest_entries))]
    if provision_status is not None:
        cmd += ["--provision-status",
                str(write_json(d, "provision_status.json", provision_status))]
    if hf_status is not None:
        cmd += ["--hf-status", str(write_json(d, "hf_status.json", hf_status))]
    if template is not None:
        cmd += ["--template", str(write_json(d, "template.json", template))]
    if notes_root is not None:
        cmd += ["--skeleton-dir", str(SKELETON_DIR),
                "--notes-root", str(notes_root)]
        if sections is not None:
            cmd += ["--note-sections", str(sections)]
        else:
            # A missing sections file must be tolerated (no template ships one
            # until its migration commit).
            cmd += ["--note-sections", str(d / "no_such_sections.md")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok(r.returncode == 0,
       f"renderer exited {r.returncode}:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def clean_model_fixtures(d: Path):
    """3 models total: 2 already on disk (skipped by the provisioner),
    1 queued this boot and successfully downloaded."""
    models = d / "models"
    dest = models / "diffusion_models" / "wan22_i2v.safetensors"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"\0" * (11 * MB))
    manifest = [("https://huggingface.co/o/r/resolve/main/wan22_i2v.safetensors",
                 str(dest), None)]
    status = {
        "mode": "walk",
        "enabled_flags": ["download_wan22"],
        "workflows_copied": ["Wan 2.2 I2V/i2v.json", "Wan Animate/anim.json"],
        "skipped": ["vae.safetensors", "te.safetensors"],
        "user_supplied": [],
    }
    hf = {"wan22_i2v.safetensors": {"status": "done", "error": None,
                                    "url": "https://huggingface.co/o/r/resolve/main/wan22_i2v.safetensors"}}
    return manifest, status, hf


# ------------------------------------------------------------------- tests

def test_clean_boot_is_compact():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        out = render(d, manifest_entries=manifest, provision_status=status,
                     hf_status=hf, template=TEMPLATE)
        ok("ComfyUI is ready   https://abc123-8188.proxy.runpod.net" in out,
           f"header/URL missing:\n{out}")
        ok("  ComfyUI    v0.34.0 (approved)" in out, out)
        ok("  Runtime    comfyui-runtime @ 28a4ca1" in out, out)
        ok("  Base       cu128-comfy0.34.0-torch2.11.0" in out, out)
        ok("  GPU        NVIDIA H200 (sm90)" in out, out)
        ok("  SageAttn   enabled (sm90, baked wheel)" in out, out)
        ok("  Models     3/3 downloaded" in out, out)
        ok("  Workflows  Wan 2.2 I2V, Wan Animate (1 set)" in out, out)
        ok("  Warnings   none" in out, out)
        # Compact means COMPACT: no expansion markers, no per-model listing.
        ok("FAILED" not in out, f"clean boot must not say FAILED:\n{out}")
        ok("WARN" not in out.replace("Warnings", ""), out)
        ok("wan22_i2v.safetensors" not in out,
           f"clean boot must not list individual models:\n{out}")
        ok(out.count("=" * 60) == 3, f"expected 3 rule lines:\n{out}")


def test_failed_model_expands_with_consequence():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        # One more queued entry whose dest never appeared: a 404.
        manifest.append(
            ("https://huggingface.co/o/r/resolve/main/qwen_image_pid.pth",
             str(d / "models" / "upscale_models" / "qwen_image_pid.pth"),
             None))
        hf["qwen_image_pid.pth"] = {
            "status": "failed",
            "error": "404 Client Error: Not Found for url: https://huggingface.co/o/r/resolve/main/qwen_image_pid.pth",
            "url": "https://huggingface.co/o/r/resolve/main/qwen_image_pid.pth",
        }
        out = render(d, manifest_entries=manifest, provision_status=status,
                     hf_status=hf, template=TEMPLATE)
        ok("  Models     3/4 downloaded, 1 FAILED" in out, out)
        ok("     FAILED  qwen_image_pid.pth  404 from huggingface.co" in out,
           out)
        # The half of the message that matters: the consequence.
        ok("Workflows using this model will error." in out, out)


def test_failed_model_without_status_entry_still_named():
    # Disk audit is the authority; the reasons file is optional detail.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, _ = clean_model_fixtures(d)
        manifest.append(("https://example.com/direct/thing.pth",
                         str(d / "models" / "detection" / "thing.pth"), None))
        out = render(d, manifest_entries=manifest, provision_status=status,
                     template=TEMPLATE)
        ok("1 FAILED" in out, out)
        ok("     FAILED  thing.pth" in out, out)
        ok("Workflows using this model will error." in out, out)


def test_sub_floor_file_counts_as_failed():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        stub = d / "models" / "loras" / "small.safetensors"
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_bytes(b"\0" * (2 * MB))  # below its 5 MB floor
        manifest.append(("https://huggingface.co/o/r/resolve/main/small.safetensors",
                         str(stub), "5"))
        out = render(d, manifest_entries=manifest, provision_status=status,
                     hf_status=hf, template=TEMPLATE)
        ok("1 FAILED" in out, out)
        ok("     FAILED  small.safetensors" in out, out)


def test_sage_unsupported_names_reason_and_consequence():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        rows = [r for r in CLEAN_STATE if r[1] not in ("sage", "sage_msg",
                                                       "gpu_name")]
        rows += [("set", "gpu_name", "NVIDIA B200"),
                 ("set", "sage", "unsupported"),
                 ("set", "sage_msg",
                  "unsupported GPU arch sm100, SageAttention off")]
        out = render(d, rows, manifest_entries=manifest,
                     provision_status=status, hf_status=hf, template=TEMPLATE)
        ok("  SageAttn   DISABLED (sm100 has no kernel; B200/B300 unsupported)"
           in out, out)
        ok("Workflows still run; generation is slower without it." in out, out)
        ok("  GPU        NVIDIA B200 (sm100)" in out, out)


def test_sage_probe_failure_says_report_it():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        rows = [r for r in CLEAN_STATE if r[1] not in ("sage", "sage_msg")]
        rows += [("set", "sage", "probe_failed"),
                 ("set", "sage_msg",
                  "SageAttention probe failed on sm90, launching without it, report this")]
        out = render(d, rows, manifest_entries=manifest,
                     provision_status=status, hf_status=hf, template=TEMPLATE)
        ok("  SageAttn   DISABLED (probe failed on sm90; this is a bug, please report it)"
           in out, out)
        ok("Workflows still run; generation is slower without it." in out, out)


def test_sage_off_for_template_is_one_calm_line():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        rows = [r for r in CLEAN_STATE if r[1] not in ("sage", "sage_msg")]
        rows += [("set", "sage", "off_template")]
        out = render(d, rows, manifest_entries=manifest,
                     provision_status=status, hf_status=hf, template=TEMPLATE)
        ok("  SageAttn   off (not used by this template)" in out, out)
        ok("DISABLED" not in out, out)


def test_warnings_expand():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        rows = list(CLEAN_STATE) + [
            ("warn", "DNS resolution failed at boot (RunPod Global Networking enabled?)"),
            ("warn", "pre_download hook returned nonzero"),
        ]
        out = render(d, rows, manifest_entries=manifest,
                     provision_status=status, hf_status=hf, template=TEMPLATE)
        ok("  Warnings   2" in out, out)
        ok("     WARN    DNS resolution failed at boot" in out, out)
        ok("     WARN    pre_download hook returned nonzero" in out, out)


def test_not_ready_header_never_claims_ready():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rows = [r for r in CLEAN_STATE if r[1] != "ready"]
        rows += [("set", "ready", "false"),
                 ("warn", "ComfyUI process exited with code 1 before port 8188 became ready after 35s")]
        out = render(d, rows)
        ok("is ready" not in out, out)
        ok("ComfyUI FAILED to start" in out, out)
        ok("process exited with code 1" in out, out)


def test_missing_pod_id_prints_port_not_a_broken_url():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        rows = [r for r in CLEAN_STATE if r[1] != "pod_id"]
        out = render(d, rows)
        ok("proxy.runpod.net" not in out, f"broken URL rendered:\n{out}")
        ok("port 8188" in out, out)


def test_provisioner_never_ran_renders_unknown_with_consequence():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        out = render(d)  # no manifest, no provision status
        ok("  Models     unknown (provisioner did not run" in out, out)
        ok("Bundled workflows may be missing their models." in out, out)
        ok("  Workflows  unknown (provisioner did not run)" in out, out)


def test_three_notes_written_as_markdownnote_workflows():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        root = d / "wf"
        out = render(d, manifest_entries=manifest, provision_status=status,
                     hf_status=hf, template=TEMPLATE, notes_root=root)
        ids = set()
        for folder, title in zip(NOTE_FOLDERS, NOTE_TITLES):
            note = root / folder / f"{title}.json"
            ok(note.is_file(), f"{title} note not written")
            doc = json.loads(note.read_text())
            for key in ("nodes", "links", "groups", "config", "extra",
                        "version", "last_node_id", "last_link_id"):
                ok(key in doc, f"{title}: workflow shape missing {key}")
            ok(len(doc["nodes"]) == 1, doc["nodes"])
            node = doc["nodes"][0]
            ok(node["type"] == "MarkdownNote", node)
            ok(node["title"] == title, node)
            ids.add(doc["id"])
        ok(len(ids) == 3, f"note workflow ids must differ: {ids}")
        # The folders sort in note order, and above workflow set folders.
        listed = sorted(p.name for p in root.iterdir())
        ok(listed == NOTE_FOLDERS, listed)
        ok(sorted(NOTE_FOLDERS + ["Wan 2.2 I2V"])[:3] == NOTE_FOLDERS,
           "notes must sort above workflow set folders")

        # Welcome: identity, links, and the basic how-to with the real UI
        # labels of the pinned frontend (Workflows tab, Run button).
        welcome = note_md(root, "Welcome")
        ok("HearmemanAI" in welcome, welcome)
        ok("civitai.red/user/HearmemanAI" in welcome, welcome)
        ok("discord.gg" in welcome, welcome)
        ok("Workflows" in welcome and "Run" in welcome, welcome)
        ok("Load Image" in welcome and "Load Video" in welcome, welcome)
        # JupyterLab auth is opt in, so the welcome note carries BOTH states:
        # what an unset JUPYTER_TOKEN means for a pod URL anyone can reach,
        # and how to turn a login on.
        ok("JUPYTER_TOKEN" in welcome, welcome)
        ok("do not post it in public" in welcome, welcome)

        # Adding models: the CivitAI guide with the canonical variables and
        # the LoRA wget how-to with the real on-pod path.
        adding = note_md(root, "Adding Models")
        ok("CIVITAI_LORAS" in adding and "CIVITAI_CHECKPOINTS" in adding,
           adding)
        ok("civitai_token" in adding, adding)
        ok("1081768,351306" in adding, adding)
        ok("/workspace/ComfyUI/models/loras" in adding, adding)

        # Troubleshooting: the live report verbatim, the missing-model steps
        # (env var first, never Jupyter), and the switched-off list.
        trouble = note_md(root, "Troubleshooting")
        ok("ComfyUI is ready" in trouble and "3/3 downloaded" in trouble,
           trouble)
        ok(out.strip().splitlines()[0] in trouble,
           "report body not injected verbatim")
        ok("A model is missing" in trouble, trouble)
        ok("JupyterLab" in trouble, trouble)
        ok("discord.gg" in trouble, trouble)
        ok("Nothing is switched off on this pod." in trouble, trouble)


def test_note_rerendered_each_boot_with_current_report():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        root = d / "wf"
        render(d, manifest_entries=manifest, provision_status=status,
               hf_status=hf, template=TEMPLATE, notes_root=root)
        first = note_md(root, "Troubleshooting")
        ok("3/3 downloaded" in first, first)
        # Next boot: one model gone missing. The note must tell THIS story.
        manifest.append(
            ("https://huggingface.co/o/r/resolve/main/gone.safetensors",
             str(d / "models" / "vae" / "gone.safetensors"), None))
        render(d, manifest_entries=manifest, provision_status=status,
               hf_status=hf, template=TEMPLATE, notes_root=root)
        second = note_md(root, "Troubleshooting")
        ok("3/4 downloaded, 1 FAILED" in second, second)
        ok("gone.safetensors" in second, second)


def test_gate_both_ways_note_written_with_zero_flags():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        status = {"mode": "walk", "enabled_flags": [],
                  "workflows_copied": [], "skipped": [], "user_supplied": []}
        root = d / "wf"
        out = render(d, manifest_entries=[], provision_status=status,
                     template=TEMPLATE, notes_root=root)
        ok("  Workflows  none (no workflow sets enabled)" in out, out)
        ok("  Models     none requested (no download flags enabled)" in out,
           out)
        for folder, title in zip(NOTE_FOLDERS, NOTE_TITLES):
            ok((root / folder / f"{title}.json").is_file(),
               f"{title} note must be written on EVERY boot, flags or not")


def test_off_entries_land_in_the_note():
    # Decision #18: switched-off features (e.g. LTX-2.5 licence gate) are
    # discoverable in the note, with how to switch them back on.
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        rows = list(CLEAN_STATE) + [
            ("off", "LTX-2.5 workflows",
             "Accept the licence on the model page, set HF_TOKEN, restart the pod."),
        ]
        root = d / "wf"
        render(d, rows, manifest_entries=manifest, provision_status=status,
               hf_status=hf, template=TEMPLATE, notes_root=root)
        md = note_md(root, "Troubleshooting")
        ok("LTX-2.5 workflows" in md, md)
        ok("Accept the licence" in md, md)
        ok("Nothing is switched off" not in md, md)


def test_template_sections_injected_when_present():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        sections = d / "note_sections.md"
        sections.write_text("## Wan specific\n\nUse the I2V workflow first.\n")
        root = d / "wf"
        render(d, manifest_entries=manifest, provision_status=status,
               hf_status=hf, template=TEMPLATE, notes_root=root,
               sections=sections)
        # Template sections land in the welcome note, nowhere else.
        ok("Use the I2V workflow first." in note_md(root, "Welcome"),
           "template sections must land in the welcome note")
        ok("Use the I2V workflow first."
           not in note_md(root, "Troubleshooting"),
           "template sections must not leak into troubleshooting")


def test_jupyter_section_follows_the_template():
    """The welcome note must not advertise a port nothing answers on.

    template.json "jupyter": false means the runtime never launches
    JupyterLab (src/start.sh:192,:207) and the client's RunPod template
    publishes 8188 only, so the "JupyterLab and your pod URL" section is a
    lie on that pod. Default and "jupyter": true keep it verbatim: the four
    live templates carry no such key.
    """
    for template_extra, want in (({}, True),
                                 ({"jupyter": True}, True),
                                 ({"jupyter": False}, False),
                                 ({"jupyter": "False"}, False),
                                 ({"jupyter": "no"}, True)):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            manifest, status, hf = clean_model_fixtures(d)
            root = d / "wf"
            render(d, manifest_entries=manifest, provision_status=status,
                   hf_status=hf, template={**TEMPLATE, **template_extra},
                   notes_root=root)
            md = note_md(root, "Welcome")
            got = "JupyterLab and your pod URL" in md
            ok(got == want,
               f"template {template_extra}: JupyterLab section present={got}, "
               f"want {want}")
            ok(("8888" in md) == want,
               f"template {template_extra}: port 8888 must only be named when "
               f"JupyterLab actually runs")
            ok(("JUPYTER_TOKEN" in md) == want,
               f"template {template_extra}: JUPYTER_TOKEN advice must only "
               f"appear when JupyterLab actually runs")
            # Whatever happens to that section, the rest of the note survives.
            ok("How to generate" in md,
               f"template {template_extra}: the rest of the note must survive")
            ok("<!--" not in md,
               f"template {template_extra}: no section marker may reach the "
               f"customer: {md!r}")


def test_no_em_or_en_dashes_anywhere():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        manifest, status, hf = clean_model_fixtures(d)
        root = d / "wf"
        out = render(d, manifest_entries=manifest, provision_status=status,
                     hf_status=hf, template=TEMPLATE, notes_root=root)
        texts = [(out, "report")]
        texts += [(note_md(root, t), f"{t} note") for t in NOTE_TITLES]
        for text, name in texts:
            ok("—" not in text, f"em dash in {name}")
            ok("–" not in text, f"en dash in {name}")


def test_end_to_end_with_the_real_provisioner():
    """The renderer's inputs come from the real provisioner, not only from
    hand-written fixtures: PROVISION_STATUS_FILE + the manifest it writes."""
    import os
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "workflows" / "Wan 2.2 I2V"
        src.mkdir(parents=True)
        (src / "i2v.json").write_text(json.dumps(
            {"nodes": [{"id": 1, "widgets_values": ["dit.safetensors"]}]}))
        write_json(d, "registry.json", {
            "dit.safetensors": {"url": "https://huggingface.co/o/r/resolve/main/dit.safetensors",
                                "subdir": "diffusion_models"}})
        template = {"provisioning_mode": "walk",
                    "flags": {"download_wan22": {"folders": ["Wan 2.2 I2V"]}}}
        write_json(d, "template.json", template)
        status_path = d / "provision_status.json"
        env = dict(os.environ)
        env.update({"download_wan22": "true",
                    "PROVISION_STATUS_FILE": str(status_path)})
        r = subprocess.run(
            [sys.executable, str(PROVISIONER),
             "--template", str(d / "template.json"),
             "--registry", str(d / "registry.json"),
             "--workflows-src", str(d / "workflows"),
             "--workflows-dst", str(d / "wf_out"),
             "--models-root", str(d / "models"),
             "--manifest", str(d / "queue.tsv")],
            capture_output=True, text=True, env=env)
        ok(r.returncode == 0, r.stdout + r.stderr)
        ok(status_path.is_file(), "provisioner wrote no status file")
        status = json.loads(status_path.read_text())
        ok(status["enabled_flags"] == ["download_wan22"], status)
        ok(status["workflows_copied"] == ["Wan 2.2 I2V/i2v.json"], status)

        # Simulate the download landing, then render from the real artifacts.
        dest = d / "models" / "diffusion_models" / "dit.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\0" * (11 * MB))
        cmd = [sys.executable, str(RENDERER),
               "--state", str(write_state(d, CLEAN_STATE)),
               "--manifest", str(d / "queue.tsv"),
               "--provision-status", str(status_path),
               "--template", str(d / "template.json")]
        rr = subprocess.run(cmd, capture_output=True, text=True)
        ok(rr.returncode == 0, rr.stdout + rr.stderr)
        ok("  Models     1/1 downloaded" in rr.stdout, rr.stdout)
        ok("  Workflows  Wan 2.2 I2V (1 set)" in rr.stdout, rr.stdout)


def test_volume_copy_row():
    """A model still staged must be reported as pending, not as durable."""
    tmpdir = tempfile.mkdtemp()
    tmp = Path(tmpdir)
    models = tmp / "models"
    models.mkdir(parents=True)
    stage = tmp / "hf_stage"
    stage.mkdir()

    landed = models / "landed.safetensors"
    landed.write_bytes(b"\0" * 2048)
    target = stage / "staged.safetensors"
    target.write_bytes(b"\0" * 4096)
    staged = models / "staged.safetensors"
    staged.symlink_to(target)
    dead = models / "dead.safetensors"
    dead.symlink_to(stage / "gone.safetensors")

    manifest = [{"url": "u", "dest": p, "floor": 1} for p in (landed, staged, dead)]
    rows = br.volume_copy_rows(manifest)
    joined = "\n".join(rows)
    ok(rows, "a staged model must produce a pending row")
    ok("1 model," in joined, f"only the live symlink counts: {joined}")
    ok("4.0KB" in joined, f"size must come from the staged file: {joined}")
    ok("safe to restart" in joined.lower(), "must tell the user when it is durable")

    ok(br.volume_copy_rows([{"url": "u", "dest": landed, "floor": 1}]) == [],
       "a fully landed model must produce no pending row")
    ok(br.volume_copy_rows(None) == [], "no manifest, no row")


def main():
    test_volume_copy_row()
    test_clean_boot_is_compact()
    test_failed_model_expands_with_consequence()
    test_failed_model_without_status_entry_still_named()
    test_sub_floor_file_counts_as_failed()
    test_sage_unsupported_names_reason_and_consequence()
    test_sage_probe_failure_says_report_it()
    test_sage_off_for_template_is_one_calm_line()
    test_warnings_expand()
    test_not_ready_header_never_claims_ready()
    test_missing_pod_id_prints_port_not_a_broken_url()
    test_provisioner_never_ran_renders_unknown_with_consequence()
    test_three_notes_written_as_markdownnote_workflows()
    test_note_rerendered_each_boot_with_current_report()
    test_gate_both_ways_note_written_with_zero_flags()
    test_off_entries_land_in_the_note()
    test_template_sections_injected_when_present()
    test_jupyter_section_follows_the_template()
    test_no_em_or_en_dashes_anywhere()
    test_end_to_end_with_the_real_provisioner()
    print(f"boot report self-test: all good ({CHECKS} assertions)")


if __name__ == "__main__":
    main()
