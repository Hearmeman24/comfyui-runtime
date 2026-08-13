# Troubleshooting

## A model is missing

If a workflow that comes with this template cannot find its model, do not
upload the model yourself through JupyterLab. That is the most common
mistake, and it does not fix the real problem. Do this instead:

1. Check the environment variable for that model set. On RunPod open your
   pod, click Edit Pod, and look at the environment variables. If the
   value is wrong, fix it and restart the pod. The models download on the
   next boot.
2. If the variable is set correctly and the model is still missing, ask
   for help in Discord: https://discord.gg/ZVWVhT43GW. Paste the "This
   boot" block from below.

## This boot

This block describes the boot that just finished. Support will ask you to
paste it.

```
{{DEPLOYMENT_REPORT}}
```

About the SageAttn line: SageAttention makes generation faster. It is on
automatically when your GPU supports it. If the line says DISABLED, your
GPU has no support for it, so it switched itself off. Everything still
works, generation is a bit slower, and there is nothing for you to fix.

## Switched off on this pod

{{DISABLED_FEATURES}}
