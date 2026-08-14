# Welcome to {{TEMPLATE_NAME}}

Hi, this is a template by HearmemanAI.

- More models and workflows: https://civitai.red/user/HearmemanAI
- Discord, for help and new releases: https://discord.gg/ZVWVhT43GW

## How to generate

1. On the left side, click Workflows and load the workflow you want to
   use.
2. If you need to add an image, click the Load Image node and upload it
   there. If you need to load a video, use the Load Video node.
3. Once everything is set, click Run and wait for the generation to
   finish.

## JupyterLab and your pod URL

JupyterLab runs on port 8888, and by default it asks for no login. Anyone
who has your pod URL can open it, reach your files and use the terminal.
Treat the URL like a password and do not post it in public.

If you want a login, add an environment variable called JUPYTER_TOKEN to
your pod and set it to any secret text you pick. Restart the pod.
JupyterLab then asks for that value before it opens. The value is never
written to the pod log, so your log stays safe to share with support.

{{TEMPLATE_SECTIONS}}
