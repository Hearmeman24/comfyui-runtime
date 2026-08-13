# {{TEMPLATE_NAME}}: read this first

## Download any CivitAI model straight into your pod

My templates can pull LoRAs and checkpoints from CivitAI for you at boot.
No downloading to your machine, no uploading to the pod.

Get your CivitAI token. You only do this once:

1. Click your profile picture, top right on CivitAI.
2. Click Account Settings, the cog icon.
3. Scroll all the way down to API Keys.
4. Click Add API Key, give it a name, and copy it somewhere safe.

Then every time you deploy:

1. Click Edit Template before you deploy, not after.
2. Expand the environment variables tab.
3. Paste your token into the civitai_token variable.
4. Add the model IDs you want to download: LoRAs in
   LORAS_IDS_TO_DOWNLOAD, checkpoints in CHECKPOINT_IDS_TO_DOWNLOAD.

The model ID is the second part of the AIR on the model page. For more
than one, separate them with commas:

```
1081768,351306
```

Save and deploy. They download on their own. Your pod is already
running, so this is how you set it up for your next deploy. You can also
edit the same variables on this pod and restart it.

If the template you are deploying has no variable for model IDs, it
means I have not built it into that one yet. I have a lot of templates.
Tell me in the help-and-support channel on Discord and I will prioritise
it.

Full write-up:
https://civitai.red/articles/12333/how-to-use-hearmemans-civitai-downloader-when-deploying-a-runpod-template

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

About the SageAttn line: SageAttention makes generation faster. It is on
automatically when your GPU supports it. If the line says DISABLED, your
GPU has no support for it, so it switched itself off. Everything still
works, generation is a bit slower, and there is nothing for you to fix.

## Switched off on this pod

{{DISABLED_FEATURES}}

{{TEMPLATE_SECTIONS}}

## Links

- More models and workflows: https://civitai.red/user/HearmemanAI
- Help and support: https://discord.gg/ZVWVhT43GW
