# {{TEMPLATE_NAME}}: read this first

## Add a LoRA from CivitAI

[PLACEHOLDER. Aviv's CivitAI download guide goes here. Do not ship this
note until this section is filled in.]

## Add a LoRA from Hugging Face

1. On huggingface.co, open the page of the LoRA you want. Click "Files and
   versions". Right click the download arrow next to the .safetensors file
   and copy the link.
2. Open a terminal on your pod: on RunPod click Connect, open port 8888
   (JupyterLab), then click Terminal.
3. In the terminal, run these two commands. Replace the link with the one
   you copied:

```
cd /workspace/ComfyUI/models/loras
wget "https://huggingface.co/username/my-lora/resolve/main/my-lora.safetensors"
```

4. When the download finishes, go back to ComfyUI and press R to refresh.
   Your LoRA now shows up in the LoRA loader's list.

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

## Switched off on this pod

{{DISABLED_FEATURES}}

{{TEMPLATE_SECTIONS}}

## Links

- More models and workflows: https://civitai.com/user/HearmemanAI
- Help and support: https://discord.gg/ZVWVhT43GW
