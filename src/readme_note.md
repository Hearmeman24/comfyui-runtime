# {{TEMPLATE_NAME}}: read this first

This note is rebuilt on every boot. The report below describes the boot that
just finished, not the day the pod was created. If something looks wrong,
restart the pod and read this note again.

## This boot

```
{{DEPLOYMENT_REPORT}}
```

## Switched off on this pod

{{DISABLED_FEATURES}}

## How this template works

- The bundled workflows are in this workflow list, grouped in folders.
- Models download on the first boot. Large sets take time. Progress is in the
  pod log, and the final result is in the report above.
- Your workflows, outputs and inputs live on the network volume. They survive
  restarts and pod re-creation on the same volume.
- Model sets are switched with environment variables on the pod. A default
  set is on unless you set its variable to false. Optional sets are off
  unless you turn them on. Restart the pod after changing a variable.
- Extra models from CivitAI: set CHECKPOINT_IDS_TO_DOWNLOAD or
  LORAS_IDS_TO_DOWNLOAD to a comma separated list of version IDs.

{{TEMPLATE_SECTIONS}}

## Troubleshooting

- A node is red or a model is missing: check the Models line in the report
  above. A failed download is named there with its reason.
- ComfyUI did not start: open the pod log. The startup log path is printed at
  the end of the boot log, and the boot log itself is saved as comfyui.log on
  your volume.
- The log says "Could not resolve host": RunPod's Global Networking setting
  is enabled on this pod. That setting removes public DNS, so nothing can
  download. Deploy a new pod with Global Networking disabled.
- Generation is slower than expected: check the SageAttn line in the report.
  On some GPUs SageAttention has no kernel and is switched off.
- Something worked yesterday and not today: compare the report above with
  what you remember. It tells you what changed on this boot.

## Help

Discord: https://discord.gg/ZVWVhT43GW

When you ask for help, paste the "This boot" block from the top of this note.
It answers the first five questions support will ask you.
