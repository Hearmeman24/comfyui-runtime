#!/usr/bin/env python3
"""SageAttention kernel probe (CONTRACTS.md section 8).

A REAL kernel launch, because import success is not evidence: the old baked
wheel imports fine on an H100 and then cannot launch. Extracted from the
heredoc duplicated at comfyui-wan/src/start.sh:26-34 and
comfyui-minimax/src/start.sh:36-44.

Invocation (no arguments):

    python3 /comfyui-runtime/src/sage_probe.py

Exit codes / stdout (exactly one line, D9 wording):
    0  kernel launch succeeded     "SageAttention probe passed on sm<XX>"
    1  supported arch, probe raised
       (import error, launch failure, no CUDA device, unreadable capability)
                                   "SageAttention probe failed on sm<XX>,
                                    launching without it, report this"
    2  arch not in the dispatch set
                                   "unsupported GPU arch sm<XX>, SageAttention off"

The arch check runs BEFORE importing sageattention, so exit 2 is reachable
even when the wheel is absent or broken.
"""

import sys

# Exactly the arms upstream core.py:143-157 dispatches at HEAD d1a57a546
# (sm80/86/89/90/120/121; spec section 2). sm86 is in the set although it needs
# no cubin (triton JIT path, plan D7). sm70 (V100) and sm100/sm103 (B200/B300)
# exit 2: upstream has no arm for them and no wheel can fix it (plan section 5).
DISPATCH_SET = {(8, 0), (8, 6), (8, 9), (9, 0), (12, 0), (12, 1)}


def main():
    try:
        import torch
        cap = torch.cuda.get_device_capability()
        arch = "%d%d" % (cap[0], cap[1])
    except Exception as exc:
        print("SageAttention probe failed on sm??, launching without it, report this")
        print("sage_probe: could not read device capability: %r" % (exc,), file=sys.stderr)
        return 1

    if cap not in DISPATCH_SET:
        print("unsupported GPU arch sm%s, SageAttention off" % arch)
        return 2

    try:
        from sageattention import sageattn
        # Frozen probe body (comfyui-wan/src/start.sh:27-33).
        q = torch.randn(1, 8, 128, 64, dtype=torch.float16, device="cuda")
        sageattn(q, q.clone(), q.clone())
        torch.cuda.synchronize()
    except Exception as exc:
        print("SageAttention probe failed on sm%s, launching without it, report this" % arch)
        print("sage_probe: %r" % (exc,), file=sys.stderr)
        return 1

    print("SageAttention probe passed on sm%s" % arch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
