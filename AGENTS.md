# Server GPU restriction

The user explicitly prohibited using physical GPUs **0 and 1** on `eeg-server`
on 2026-11-05 because they belong to other users. Do not allocate, train,
qualify, evaluate, or register a queue targeting either device.

Protected UUIDs:

- GPU 0: `GPU-bb3b892e-454f-e10e-c305-5ab94db2c384`
- GPU 1: `GPU-8b2cacdf-4470-21f2-01c5-e15475dbdfae`

Only physical GPUs 2–7 are eligible, subject to actual resource admission.
Never terminate another user's processes. Resolve physical UUIDs before setting
`CUDA_VISIBLE_DEVICES`; the logical `cuda:0` inside a restricted process does
not identify physical GPU 0.

Historical plans and qualification records mentioning GPU 1 are provenance,
not permission to use it again. The native IK continuation's old GPU 1
qualification must not be relabeled as qualification for another device.
Complete the original B10096 qualification on an allowed device before its
new full execution, while retaining the historical 68 batches and their actual
producer identities unchanged.
