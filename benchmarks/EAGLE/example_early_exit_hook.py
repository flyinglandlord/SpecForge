"""
Example script demonstrating how to use eagenerate_with_early_exit_hook
for evaluating MTP module early exit behavior.
"""

import torch
import numpy as np
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def simple_early_exit_hook(hook_state):
    """
    Simple example hook that predicts accept length based on logit confidence.
    
    This is a placeholder implementation. Replace with your actual MTP predictor.
    
    Args:
        hook_state: Dictionary containing:
            - logits: verification logits
            - candidates: candidate tokens
            - hidden_state: hidden states
            - outputs: model outputs
            - step: current decoding step
            - input_ids: current input token ids
            - best_candidate: index of best candidate
            
    Returns:
        predicted_accept_length: integer prediction of how many tokens will be accepted
    """
    logits = hook_state['mtp_module_logits']
    candidates = hook_state['draft_candidates']
    retrieve_indices = hook_state['retrieve_indices']
    
    # Example strategy: predict based on maximum probability confidence
    # This is a simple heuristic - replace with your actual MTP model
    probs = torch.softmax(logits, dim=-1)
    max_probs = torch.max(probs, dim=-1)[0]
    
    # Count consecutive positions with high confidence (> 0.5 threshold)
    threshold = 0.5
    predicted_length = 0
    for i in range(len(max_probs)):
        if max_probs[i] > threshold and candidates[best_candidate, i+1] != -1:
            predicted_length += 1
        else:
            break
    
    return predicted_length


def confidence_based_hook(threshold=0.0):
    """
    Returns a hook function with configurable confidence threshold.
    
    Args:
        threshold: minimum probability threshold for accepting tokens
    """
    def hook(hook_state):
        logits = hook_state['mtp_module_logits']
        candidates = hook_state['draft_candidates'][0]
        retrieve_indices = hook_state['retrieve_indices']

        prob = torch.softmax(logits, dim=-1)

        predicted_length = 0
        for i, token in enumerate(candidates[1:]):
            if token == -1:  # padding token
                break
            token_prob, _ = torch.max(prob[i], dim=-1)
            if token_prob > threshold:
                predicted_length += 1
            else:
                break
        return predicted_length
    
    return hook


def entropy_based_hook(max_entropy=1.0):
    """
    Hook that uses entropy of probability distribution to predict acceptance.
    Lower entropy (more confident) = higher acceptance.
    
    Args:
        max_entropy: maximum entropy threshold for accepting tokens
    """
    def hook(hook_state):
        sample_token = hook_state['p_token']  # [B, V]

        entropy = -torch.sum(
            sample_token * torch.log(sample_token + 1e-10),
            dim=-1
        )  # [B]

        # hyper-params
        L_min = 0
        L_max = 8

        # empirical entropy range
        H_min = 0.0
        H_max = max_entropy   # 对应 top-k / 温度采样后一个合理上界，可调

        # normalize entropy to [0, 1]
        entropy_norm = (entropy - H_min) / (H_max - H_min)
        entropy_norm = entropy_norm.clamp(0.0, 1.0)

        # inverse mapping: high entropy -> small length
        predicted_length = L_max - entropy_norm * (L_max - L_min)

        # 如果你需要整数长度
        predicted_length = predicted_length.round().long()

        return predicted_length
    
    return hook


def learned_mtp_hook(mtp_model):
    """
    Hook that uses a learned MTP (Multi-Token Prediction) model.
    
    Args:
        mtp_model: A trained model that predicts accept length
                   Should have a method: predict(hidden_state, logits) -> int
    """
    def hook(hook_state):
        hidden_state = hook_state['hidden_state']
        logits = hook_state['logits']
        
        # Use your trained MTP model to predict accept length
        with torch.no_grad():
            predicted_length = mtp_model.predict(hidden_state, logits)
        
        return predicted_length
    
    return hook

def build_token_level_info(
    input_ids,
    output_ids,
    stats,
    tokenizer,
):
    """
    Strictly align output tokens with true_accept_lengths.

    Each decoding step t produces:
        true_accept_length[t] draft tokens
        + 1 verification token

    Returns:
        token_infos: List[dict]
    """

    input_len = input_ids.shape[1]
    gen_ids = output_ids[0][input_len:]
    gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids)

    true_accept_lengths = stats["true_accept_lengths"]

    token_infos = []
    cursor = 0  # pointer over generated tokens

    for step_idx, acc_len in enumerate(true_accept_lengths):
        # verification token (always exists)
        if cursor >= len(gen_tokens):
            raise RuntimeError("Cursor overflow when processing verification token")

        token_infos.append({
            "token": gen_tokens[cursor].replace("▁", " "),
            "step_idx": step_idx,
            "is_draft": False,
            "accept_len": acc_len,
        })
        cursor += 1

        # draft tokens
        for i in range(acc_len):
            if cursor >= len(gen_tokens):
                raise RuntimeError("Cursor overflow when processing draft tokens")

            token_infos.append({
                "token": gen_tokens[cursor].replace("▁", " "),
                "step_idx": step_idx,
                "is_draft": True,
                "accept_len": acc_len,
            })
            cursor += 1

    # ✅ consistency check（你点名要的）
    assert cursor == len(gen_tokens), (
        f"Token alignment mismatch: "
        f"sum(true_accept_length + 1) = {cursor}, "
        f"but output tokens = {len(gen_tokens)}"
    )

    return token_infos

import re

def clean_token(token: str) -> str:
    """
    Clean tokenizer artifacts for visualization.
    """
    # sentencepiece / BPE 空格标记
    token = token.replace("▁", "")
    token = token.replace("Ġ", "")

    # 去掉不可见控制字符
    token = re.sub(r"[\x00-\x1F\x7F]", "", token)

    # 空 token 给一个占位
    if token == "":
        token = "␣"

    return token

from matplotlib.patches import Rectangle
import math


def plot_token_acceptance(
    token_infos,
    max_tokens_per_row=20,
    cmap_name="Blues",
    draft_color="#E0E0E0",
    cell_width=1.0,
    cell_height=1.0,
    font_size=10,
):
    """
    Grid-aligned token visualization.
    Each token occupies a fixed-size cell (no overlap).
    """

    tokens = [clean_token(x["token"]) for x in token_infos]
    accept_lens = [x["accept_len"] for x in token_infos]
    is_draft = [x["is_draft"] for x in token_infos]

    max_accept = max(accept_lens) + 1e-5
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=max_accept)

    num_tokens = len(tokens)
    num_rows = math.ceil(num_tokens / max_tokens_per_row)

    fig_width = max_tokens_per_row * cell_width * 0.6
    fig_height = num_rows * cell_height * 0.6 + 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    for idx, (tok, acc, draft) in enumerate(zip(tokens, accept_lens, is_draft)):
        row = idx // max_tokens_per_row
        col = idx % max_tokens_per_row

        x = col * cell_width
        y = -row * cell_height

        if draft:
            face_color = draft_color
        else:
            face_color = cmap(norm(acc))

        # 1️⃣ 画矩形（严格对齐）
        rect = Rectangle(
            (x, y),
            cell_width,
            cell_height,
            linewidth=0.5,
            edgecolor="white",
            facecolor=face_color,
        )
        ax.add_patch(rect)

        # 2️⃣ 画文字（居中，黑字）
        ax.text(
            x + cell_width / 2,
            y + cell_height / 2,
            tok,
            ha="center",
            va="center",
            fontsize=font_size,
            color="black",
            clip_on=True,
        )

    ax.set_xlim(0, max_tokens_per_row * cell_width)
    ax.set_ylim(-num_rows * cell_height, cell_height * 0.2)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02)
    cbar.set_label("True Accept Length")

    plt.show()
    plt.savefig("token_acceptance_visualization.png", bbox_inches="tight")


def run_evaluation_example():
    """
    Example of how to run evaluation with early exit hook.
    """
    # Load your model
    depth = 7
    base_model_path = "/mtc/models/qwen3-8b"
    EAGLE_model_path = "/mtc/models/qwen3-8b-eagle3"
    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=EAGLE_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=depth+1,
        depth=depth,
        top_k=1,
    )
    model.eval()

    # Example input
    your_message="Complete the code I provided.from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n"
    conv = get_conversation_template("qwen3")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
    output=model.tokenizer.decode(output_ids[0])
    
    # Option 1: Use simple confidence-based hook
    hook = confidence_based_hook(threshold=0.3)
    
    # Option 2: Use entropy-based hook
    # hook = entropy_based_hook(max_entropy=0.8)
    
    # Option 3: Use your own custom hook
    # hook = simple_early_exit_hook
    
    # Generate with early exit hook and collect statistics
    output_ids, stats = model.eagenerate_with_early_exit_hook(
        input_ids=input_ids,
        log=True,  # Important: set log=True to get statistics
        max_new_tokens=1024,
        top_k=1,
        early_exit_hook=hook,
    )

    # Based on the output_ids and input_ids and stats, draw a heat graph to highlight the accepted length vs. tokens
    # For example, the draft tokens are colored in blue, and the other tokens are colored based on the accepted length on this position.
    # === Token-level visualization ===
    token_infos = build_token_level_info(
        input_ids=input_ids,
        output_ids=output_ids,
        stats=stats,
        tokenizer=model.tokenizer,
    )

    plot_token_acceptance(
        token_infos,
        max_tokens_per_row=15,   # 控制一行多少 token
        cell_width=3.0,
        cell_height=1.0,
    )
    
    # Print evaluation results
    print("=" * 80)
    print("Early Exit Hook Evaluation Results")
    print("=" * 80)
    print(f"Total decoding steps: {stats['total_steps']}")
    print(f"Total tokens generated: {stats['total_tokens']}")
    print(f"\nAcceptance Statistics:")
    print(f"  Average true accept length: {stats['avg_true_accept_length']:.2f}")
    print(f"  Average predicted accept length: {stats['avg_predicted_accept_length']:.2f}")
    print(f"  Average length difference: {stats['avg_length_difference']:.2f}")
    print(f"\nAcceptance Rates:")
    print(f"  True acceptance rate: {stats['true_acceptance_rate']:.4f}")
    print(f"  Predicted acceptance rate: {stats['predicted_acceptance_rate']:.4f}")
    print(f"  Acceptance rate gap: {stats['acceptance_rate_gap']:.4f}")
    print(f"\nEarly Exit Statistics:")
    print(f"  Early exit rate: {stats['early_exit_rate']:.2%}")
    print(f"  (Percentage of steps where prediction < true length)")
    
    # Detailed per-step analysis
    print(f"\nPer-Step Analysis (first 10 steps):")
    for i in range(min(10, len(stats['true_accept_lengths']))):
        true_len = stats['true_accept_lengths'][i]
        pred_len = stats['predicted_accept_lengths'][i]
        early_exit = stats['early_exit_triggered'][i]
        print(f"  Step {i+1}: True={true_len}, Pred={pred_len}, "
              f"Diff={true_len-pred_len}, EarlyExit={early_exit}")
    
    # print("Example script loaded. Uncomment the code above to run actual evaluation.")
    # print("\nKey statistics you'll get:")
    # print("  - avg_true_accept_length: Average number of tokens actually accepted")
    # print("  - avg_predicted_accept_length: Average predicted by your hook")
    # print("  - true_acceptance_rate: Actual acceptance rate")
    # print("  - predicted_acceptance_rate: Predicted acceptance rate")
    # print("  - acceptance_rate_gap: |predicted - true|")
    # print("  - early_exit_rate: How often prediction underestimates truth")
    # print("  - avg_length_difference: Average (true - predicted)")


if __name__ == "__main__":
    run_evaluation_example()
