import os
import json
import argparse
import torch
import numpy as np
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from eagle.model.ea_model import EaModel


# =========================================================
# Early-exit hooks
# =========================================================
def early_exit_until_zero(hook_state):
    """
    Accept tokens sequentially until token id == 0 is encountered.
    """
    candidates = hook_state["draft_candidates"]  # [1, K+1]
    accept_len = 0
    for tok in candidates[0]:
        if tok.item() == 0:
            break
        accept_len += 1
    return max(accept_len - 1, 0)


def fully_exit_hook(hook_state):
    """
    Baseline: verify all draft tokens.
    """
    candidates = hook_state["draft_candidates"]
    return len(candidates[0]) - 1


# =========================================================
# Prompt builders
# =========================================================
def build_prompt(sample, dataset):
    if dataset == "humaneval":
        return [
            {
                "role": "system",
                "content": "You are an AI programming assistant. Complete the following Python function."
            },
            {"role": "user", "content": sample["prompt"]},
        ]

    if dataset == "gsm8k":
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves math problems."
            },
            {"role": "user", "content": sample["question"]},
        ]

    if dataset == "alpaca":
        return [
            {"role": "user",
             "content": sample["instruction"]
             + ("\n\n" + sample["input"] if sample["input"].strip() else "")
            }
        ]

    if dataset == "mt_bench":
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": sample["prompt"][0]},
        ]



# =========================================================
# Single-sample evaluation
# =========================================================
@torch.no_grad()
def run_one_sample(
    messages,
    tokenizer,
    baseline_model,
    dynamic_model,
    device_baseline,
    device_dynamic,
    max_new_tokens=2048,
):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt")

    # ===== baseline =====
    baseline_output_ids, stats_baseline = baseline_model.eagenerate_with_early_exit_hook(
        input_ids=inputs["input_ids"].to(device_baseline),
        log=True,
        max_new_tokens=max_new_tokens,
        top_k=1,
        early_exit_hook=fully_exit_hook,
    )

    # ===== dynamic =====
    dynamic_output_ids, stats_dynamic = dynamic_model.eagenerate_with_early_exit_hook(
        input_ids=inputs["input_ids"].to(device_dynamic),
        log=True,
        max_new_tokens=max_new_tokens,
        top_k=1,
        early_exit_hook=early_exit_until_zero,
    )

    b_ids = baseline_output_ids[0].tolist()
    d_ids = dynamic_output_ids[0].tolist()

    stats_baseline['output'] = tokenizer.decode(b_ids, skip_special_tokens=False)
    stats_dynamic['output'] = tokenizer.decode(d_ids, skip_special_tokens=False)

    return stats_baseline, stats_dynamic


# =========================================================
# Dataset evaluation
# =========================================================
def evaluate_dataset(
    name,
    dataset,
    tokenizer,
    baseline_model,
    dynamic_model,
    device_baseline,
    device_dynamic,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}_{time.time()}.jsonl")

    # aggregate
    dyn_true, dyn_pred = [], []
    base_true, base_pred = [], []
    total_tokens = 0

    # dataset = dataset.select(range(2))

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(dataset, desc=name)):
            messages = build_prompt(sample, name)

            stats_b, stats_d = run_one_sample(
                messages,
                tokenizer,
                baseline_model,
                dynamic_model,
                device_baseline,
                device_dynamic,
            )

            dyn_true.extend(stats_d["true_accept_lengths"])
            dyn_pred.extend(stats_d["predicted_accept_lengths"])
            base_true.extend(stats_b["true_accept_lengths"])
            base_pred.extend(stats_b["predicted_accept_lengths"])

            total_tokens += stats_d["total_tokens"]

            f.write(json.dumps({
                "sample_id": i,
                "prompt": messages,
                "baseline": {
                    "true_accept_lengths": stats_b["true_accept_lengths"],
                    "predicted_accept_lengths": stats_b["predicted_accept_lengths"],
                    "output": stats_b["output"],
                },
                "dynamic": {
                    "true_accept_lengths": stats_d["true_accept_lengths"],
                    "predicted_accept_lengths": stats_d["predicted_accept_lengths"],
                    "output": stats_d["output"],
                },
            }, ensure_ascii=False) + "\n")

    # ===== 指标（严格复刻你原始代码）=====
    dyn_true = np.array(dyn_true)
    dyn_pred = np.array(dyn_pred)
    base_true = np.array(base_true)
    base_pred = np.array(base_pred)

    metrics = {
        "avg_accepted_length_dynamic": float(np.mean(dyn_true)),
        "avg_accepted_length_baseline": float(np.mean(base_true)),

        "avg_extra_verify_tokens_dynamic":
            float(np.mean(dyn_pred - dyn_true)),
        "avg_extra_verify_tokens_baseline":
            float(np.mean(base_pred - base_true)),

        "avg_saved_verified_tokens_dynamic_vs_baseline":
            float((dyn_pred.sum() - base_pred.sum()) / max(total_tokens, 1)),
    }

    return metrics

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["gsm8k", "humaneval", "alpaca", "mt_bench"])
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--baseline_device", default="cuda:2")
    parser.add_argument("--dynamic_device", default="cuda:1")
    args = parser.parse_args()

    # load dataset
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")

    elif args.dataset == "humaneval":
        dataset = load_dataset("evalplus/humanevalplus", split="test")

    elif args.dataset == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")

    elif args.dataset == "mt_bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")

    base_model_path = "/mtc/models/qwen3-8b"
    baseline_eagle_path = "/mtc/models/qwen3-8b-eagle3"
    dynamic_eagle_path = "/mtc/chenjunyi1/project/SpecForge/outputs/qwen3-8b-eagle3-idk-dynamic-sharegpt/epoch_9_step_603370"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    depth = 7

    # baseline_model = None
    # dynamic_model = None

    baseline_model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=baseline_eagle_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.baseline_device,
        total_token=depth+1,
        depth=depth,
        top_k=1,
    ).eval()

    dynamic_model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=dynamic_eagle_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.dynamic_device,
        total_token=depth+1,
        depth=depth,
        top_k=1,
    ).eval()

    metrics = evaluate_dataset(
        args.dataset,
        dataset,
        tokenizer,
        baseline_model,
        dynamic_model,
        args.baseline_device,
        args.dynamic_device,
        args.output_dir,
    )

    print(f"\n===== {args.dataset} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
