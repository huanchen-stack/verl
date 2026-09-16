# RUN 2026-09-16: Megatron TP1 replication (mg20)

Decision 10 reversed: every RL-step number below trains with `run_fullstep.sh` `TRAINER=megatron`
(Megatron-Core 0.18 via Megatron-Bridge 0.5, LoRA 16/16 through `model.lora.*`, TP=PP=1, full uniform
recompute), one A100-80GB per arm, vLLM at gmem 0.5. All arms: GSM8K (`gsm8k_messages_2048`), B = 8 x 4 = 32
requests/step, response cap 16384, 20 steps, seed 42 (second seed 43 where noted), `data.shuffle=False`
so every arm sees the same prompts per step. Steps 1-5 are warm-up (TE/JIT compiles) and excluded.
`ours` = the fused LoRA fast path + dual stream (recipe default); `Punica` = the vanilla vLLM LoRA
backend (`lora_fast_path=false lora_dual_stream=false`). Run dirs: `/data/huanchen/ps_runs/mg20_*`.

## Trainer acceptance: downstream time is linear in tokens

Per-token costs are identical across all arms: old_log_prob 0.183 ms/token, ref 0.172, update_actor
0.505 (9B). Over every arm, steps 6-20:

| model | arms | steps | slope (s/token) | intercept | R² all steps | R² excl. recompile steps |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 11 | 165 | 8.67e-4 | 5.7 s | 0.884 (23 outliers) | **0.9988**, max residual 5.7 s |
| Qwen3.5-4B | 4 | 60 | 6.04e-4 | 4.8 s | 0.965 (4 outliers) | **0.9992**, max residual 1.4 s |

The 9B slope and intercept match the archived Megatron validation fit (8.78e-4 s/token, 12.5 s). The
outliers are sporadic steps where the LoRA forward/backward (old_log_prob and update_actor, never the
ref pass) run 20-80 % slower, concentrated in steps 6-8 and steps that set a new maximum token count,
i.e. shape-triggered recompiles; they are not token-dependent and are kept in the step numbers below.
Contrast: the discarded FSDP2 runs had an intercept of 111 s from the CPU-offloaded actor.

## Qwen3.5-9B
| arm (qwen3_5_9b, Megatron TP1, B32, cap 16k, steps 6-20) | runs×steps | rollout | downstream | step | token infl. | reward | cap hits/step | switch |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BF16, Punica (baseline) | 2×15 | 1.00x (246±5 s) | 1.00x (142±6 s) | 1.00x (388±10 s) | +0% | 0.816±0.018 | 2.3 |  |
| uniform W4, Punica | 2×15 | 1.24x (198±3 s) | 0.79x (180±7 s) | 1.02x (378±10 s) | +28% | 0.742±0.015 | 5.5 |  |
| uniform W4, ours | 1×15 | 1.33x (185±5 s) | 0.81x (174±11 s) | 1.08x (359±15 s) | +24% | 0.750±0.023 | 5.1 |  |
| fixed K=5000, ours | 2×15 | 1.28x (193±5 s) | 0.92x (155±8 s) | 1.12x (348±12 s) | +7% | 0.800±0.018 | 3.4 | switch 5000-5000 (last 5000), live 11; switch 5000-5000 (last 5000), live 10 |
| live EMA slope 0, ours | 1×15 | 1.34x (184±7 s) | 0.85x (167±11 s) | 1.10x (351±17 s) | +15% | 0.777±0.027 | 4.5 | switch 1000-2250 (last 1750), live 20 |
| live EMA fitted slope, ours | 2×15 | 1.27x (194±6 s) | 0.98x (145±7 s) | 1.14x (339±12 s) | +0% | 0.798±0.019 | 2.5 | switch 5750-8250 (last 5750), live 8; switch 5750-9000 (last 9000), live 5 |
| live EMA fitted slope + INT4 penalty 2e-4, ours | 1×15 | 1.22x (202±8 s) | 1.01x (141±9 s) | 1.13x (342±16 s) | -1% | 0.787±0.027 | 2.5 | switch 8250-9000 (last 9000), live 4 |

## Qwen3.5-4B

| arm (qwen3_5_4b, Megatron TP1, B32, cap 16k, steps 6-20) | runs×steps | rollout | downstream | step | token infl. | reward | cap hits/step | switch |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BF16, Punica (baseline) | 1×15 | 1.00x (188±6 s) | 1.00x (103±6 s) | 1.00x (291±11 s) | +0% | 0.796±0.020 | 2.5 |  |
| uniform W4, Punica | 1×15 | 1.20x (157±6 s) | 0.84x (123±10 s) | 1.04x (280±15 s) | +19% | 0.798±0.036 | 3.7 |  |
| live EMA fitted slope, ours | 1×15 | 1.24x (151±6 s) | 1.00x (103±6 s) | 1.14x (254±12 s) | -0% | 0.806±0.028 | 2.5 | switch 8250-8250 (last 8250), live 6 |
| live EMA fitted slope + INT4 penalty 2e-4, ours | 1×15 | 1.17x (161±7 s) | 1.02x (101±5 s) | 1.11x (262±12 s) | -1% | 0.812±0.024 | 2.7 | switch 8250-14500 (last 10750), live 4 |

Columns: speedup = BF16-Punica time / arm time (mean over steps 6-20 of `timing_s/gen`, `timing_s/step -
timing_s/gen`, `timing_s/step`); token inflation = tokens per step vs the baseline; cap hits = responses
per step that reached 16384 tokens; switch = committed frontier range over the run (last value) and mean
live requests at the switch. `live EMA fitted slope` = `continuous_ema.sh` with
`--downstream-slope 0.000878` (9B, archived Megatron fit; the refit above gives 8.67e-4) and `0.00067`
(4B, first-pass fit; refit 6.0e-4), alpha 0.2, calibrated from the `ema{9,4}b_calib_*` rollout-only
traces and the `qwen35_{9b,4b}_ours_fullgrid` heatmaps.

## Reading

* Uniform W4 does not pay for itself end-to-end: 1.24x rollout on Punica but +28 % tokens, so the
  training side is 0.79x and the step is 1.02x; reward drops 0.816 -> 0.742.
* The live scheduler with the fitted slope keeps the rollout gain (1.27x), removes the inflation (0 %),
  and lands at 1.14x per step with reward within one SE of BF16 (0.798 vs 0.816). Its switch settles at
  5750 (seed 42) or 8250-9000 (seed 43): the optimum moves with the observed length distribution.
* The INT4 penalty (2e-4 s/token) pushes the switch to 9000 for a 1.13x step; fixed K=5000 gives 1.12x
  with +7 % tokens. Slope 0 (rollout-only objective) switches at 1000-2250, keeps 15 % inflation and
  gives 1.10x; the downstream term is what makes the scheduler pick the late switch.
* 4B: the same picture at 1.14x (EMA) vs 1.04x (uniform W4).
* Phi-4-mini has no Megatron-Bridge mapping and is not in this replication.
