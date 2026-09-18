# Deprecated: every FSDP2-trained RL-step result from this host

Decision 10 was reversed on 2026-09-16: the RL-step trainer is Megatron TP1
(`run_fullstep.sh` `TRAINER=megatron`, the branch default). Every RL-step number
measured on the RTX PRO 6000 box before that took effect here was trained with
FSDP2 and is therefore not comparable to anything the project now reports. On
2026-09-17 the user deprecated all of it, for every model.

Removed from the tree in this commit (still in git history at `c00976f7`):

- `BASELINE_2026-09-16_pro6000_matrix.md`, the 2 x 3 baseline
  ({Qwen3.5-9B, Phi-4-mini-reasoning} x {bf16, w4a16, nvfp4}), and its six
  per-step metric streams under `baselines/matrix_2026-09-16/`.

Deprecated on disk on the same day: the baseline matrix run directories, the
partial W4A16 online-EMA run that was started against that baseline and killed
at step 10, and every figure derived from those runs (rollout survival,
stopping points, token inflation, the slide deck).

Not deprecated, because no trainer was involved: the TPOT heatmaps (standalone
vLLM engines) and the 128-request EMA calibration rollouts (`rollout_only`).

Megatron is not installed in this host's environment as of 2026-09-17
(`megatron.core`, `transformer_engine`, `apex`, `megatron.bridge` all absent),
which is why both sessions on this box had pinned FSDP2. Installing the
`megatron` extra is the prerequisite for any new RL-step number here.
