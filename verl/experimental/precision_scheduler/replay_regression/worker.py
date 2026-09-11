# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Replay-regression worker: time old-logprob / reference / PPO-update on token workloads (no rollout engine).

One ``TrainingWorker`` (Megatron, TP=1) is created through Ray on a single GPU; for every workload
point the worker runs actor old-logprob (with entropy), reference logprob (no adapter) and one PPO
mini-batch update, exactly like a full RL step minus generation, and records the seconds.  The
per-point rows (``inference_s``, ``training_s``, ``total_tokens``...) feed
``downstream_regression.fit_replay_points``.  The first point is a warmup and is flagged as such.

This is a GPU tool; it is not exercised by the CPU test-suite (only ``build_batch`` is).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path
from typing import Any

DEFAULT_TARGETS = (
    "language_model.decoder.layers.*.self_attention.linear_qkv",
    "language_model.decoder.layers.*.self_attention.linear_proj",
    "language_model.decoder.layers.*.self_attention.in_proj",
    "language_model.decoder.layers.*.self_attention.out_proj",
    "language_model.decoder.layers.*.mlp.linear_fc1",
    "language_model.decoder.layers.*.mlp.linear_fc2",
)


def build_batch(samples: list[dict[str, Any]], pad_id: int):
    """Left-pad prompts, right-pad responses and assemble the DataProto a training worker expects."""
    import torch

    from verl import DataProto
    from verl.utils.model import compute_position_id_with_mask

    prompt_max = max(len(x["prompt"]) for x in samples)
    response_max = max(len(x["response"]) for x in samples)
    ids, masks, prompts, responses, response_masks = [], [], [], [], []
    for x in samples:
        p, r = x["prompt"], x["response"]
        lp, rr = prompt_max - len(p), response_max - len(r)
        prompts.append([pad_id] * lp + p)
        responses.append(r + [pad_id] * rr)
        ids.append([pad_id] * lp + p + r + [pad_id] * rr)
        masks.append([0] * lp + [1] * (len(p) + len(r)) + [0] * rr)
        response_masks.append([1] * len(r) + [0] * rr)
    attention_mask = torch.tensor(masks, dtype=torch.long)
    return DataProto.from_single_dict(
        {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "prompts": torch.tensor(prompts, dtype=torch.long),
            "responses": torch.tensor(responses, dtype=torch.long),
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "response_mask": torch.tensor(response_masks, dtype=torch.long),
        },
        meta_info={"temperature": 1.0},
    )


def _infer(wg, batch, *, no_lora: bool, entropy: bool):
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    td = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(td, calculate_entropy=entropy, compute_loss=False, no_lora_adapter=no_lora)
    started = time.perf_counter()
    out = wg.infer_batch(td).get()
    elapsed = time.perf_counter() - started
    return elapsed, no_padding_2_padding(tu.get(out, "log_probs"), td).float()


def _train(wg, batch, global_batch_size: int):
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    td = left_right_2_no_padding(batch.to_tensordict())
    tu.assign_non_tensor(
        td,
        global_batch_size=global_batch_size,
        mini_batch_size=global_batch_size,
        epochs=1,
        seed=42,
        dataloader_kwargs={"shuffle": False},
        compute_loss=True,
    )
    started = time.perf_counter()
    out = wg.train_mini_batch(td).get()
    return time.perf_counter() - started, tu.get(out, "metrics")


def make_worker_group(args: argparse.Namespace):
    import ray

    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.config.actor import ActorConfig
    from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
    from verl.workers.utils.losses import ppo_loss

    targets = list(args.targets)
    model = HFModelConfig(
        path=args.model,
        use_remove_padding=True,
        use_fused_kernels=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora={"rank": args.lora_rank, "alpha": args.lora_alpha, "target_modules": targets, "merge": False},
    )
    engine = McoreEngineConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        use_mbridge=True,
        vanilla_mbridge=False,
        dtype="bfloat16",
        param_offload=False,
        grad_offload=False,
        optimizer_offload=False,
        use_dynamic_bsz=True,
        max_token_len_per_gpu=args.token_budget,
        infer_max_token_len_per_gpu=args.token_budget,
        micro_batch_size_per_gpu=1,
        infer_micro_batch_size_per_gpu=1,
        use_fused_kernels=True,
        use_remove_padding=True,
        override_transformer_config={
            "attention_backend": "auto",
            "gradient_accumulation_fusion": False,
            "recompute_granularity": "full",
            "recompute_method": "uniform",
            "recompute_num_layers": 1,
        },
    )
    optim = McoreOptimizerConfig(lr=1e-6, total_training_steps=100, lr_decay_steps=100)
    cfg = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model,
        engine_config=engine,
        optimizer_config=optim,
        checkpoint_config=None,
    )
    pool = RayResourcePool(process_on_nodes=[1], name_prefix=f"replay_{os.getpid()}", max_colocate_count=4)
    wg = RayWorkerGroup(pool, RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=cfg))
    wg.reset()
    actor_cfg = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_mini_batch_size=128,
        ppo_micro_batch_size_per_gpu=1,
        use_dynamic_bsz=True,
        ppo_max_token_len_per_gpu=args.token_budget,
        use_kl_loss=True,
        kl_loss_coef=0.01,
        kl_loss_type="low_var_kl",
        ppo_epochs=1,
        calculate_entropy=True,
        use_torch_compile=False,
    )
    wg.set_loss_fn(partial(ppo_loss, config=actor_cfg))
    return wg, model


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    import ray
    import torch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ray.init(num_gpus=1, num_cpus=args.num_cpus, include_dashboard=False, _temp_dir=str(args.ray_temp_dir))
    try:
        wg, model = make_worker_group(args)
        manifest = json.loads((args.workloads / "manifest.json").read_text())
        by_index = {int(x["point"]): x for x in manifest["points"]}
        pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
        results = []
        for order, index in enumerate(args.indices):
            meta = json.loads(Path(by_index[index]["path"]).read_text())
            batch = build_batch(meta["samples"], pad_id)
            # Match full RL steps: actor old-logprob computes entropy, reference logprob does not.
            old_s, old = _infer(wg, batch, no_lora=False, entropy=True)
            ref_s, ref = _infer(wg, batch, no_lora=True, entropy=False)
            batch.batch["old_log_probs"] = old
            batch.batch["ref_log_prob"] = ref
            batch.batch["advantages"] = torch.ones_like(old)
            train_s, _metrics = _train(wg, batch, len(batch))
            row = {k: meta[k] for k in ("point", "batch_size", "cap", "total_tokens", "response_tokens")}
            row.update(
                {
                    "order": order,
                    "warmup": order == 0,
                    "old_logprob_s": old_s,
                    "ref_logprob_s": ref_s,
                    "inference_s": old_s + ref_s,
                    "training_s": train_s,
                    "max_seq_len": max(len(x["prompt"]) + len(x["response"]) for x in meta["samples"]),
                    "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "max_memory_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
            results.append(row)
            args.output.write_text(json.dumps(results, indent=2))
            print("REPLAY_POINT " + json.dumps(row), flush=True)
        return results
    finally:
        ray.shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint path")
    parser.add_argument("--workloads", type=Path, required=True, help="directory with manifest.json and point_*.json")
    parser.add_argument("--indices", type=lambda s: [int(x) for x in s.split(",") if x], required=True)
    parser.add_argument("--output", type=Path, required=True, help="per-GPU JSON results (one list of rows)")
    parser.add_argument("--token-budget", type=int, default=26624)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS), help="Megatron LoRA target patterns")
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--ray-temp-dir", type=Path, default=Path(f"/tmp/replay_reg_{os.getpid()}"))
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
