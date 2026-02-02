import os
import torch
import numpy as np
from eagle.model.ea_model import EaModel
from fastchat.model import get_conversation_template
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np

def mtp_sampling_early_exit_hook(hook_state):
    """
    Stochastic early exit based on MTP UNK-head logits.

    mtp_logits shape: [K, vocab_size + 1]
    The last column (-1) corresponds to exit / UNK probability.
    """
    mtp_logits = hook_state['mtp_module_logits']
    candidates = hook_state['draft_candidates']

    K = mtp_logits.shape[0]

    # take UNK / exit logits: [K]
    p_exit = torch.sigmoid(mtp_logits[:, -1])

    # sequential Bernoulli stopping
    for i in range(K):
        r = torch.rand((), device=p_exit.device)
        if r < p_exit[i]:
            return i  # accept i tokens, exit here

    # no exit happened
    return K


def baseline_sampling_early_exit_hook(hook_state):
    """
    Baseline stochastic early exit based on max softmax probability.

    mtp_logits shape: [K, vocab_size]
    At each position i:
        p_continue = max softmax prob
        p_exit = 1 - p_continue
    """
    mtp_logits = hook_state['mtp_module_logits']
    candidates = hook_state['draft_candidates']

    # number of draft tokens
    K = candidates.shape[1] - 1

    # mtp_logits: [K, V]
    # compute p_continue = max softmax prob
    probs = torch.softmax(mtp_logits, dim=-1)      # [K, V]
    p_continue = probs.max(dim=-1).values               # [K]
    # print(p_continue)

    # sequential sampling
    for i in range(K):
        r = torch.rand((), device=p_continue.device)
        if r > p_continue[i]:
            return i  # exit here, accept i tokens

    # no exit happened
    return K


def early_exit_until_zero(hook_state):
    """
    Sequential early-exit rule:
    Accept tokens from left to right until token id == 0 is encountered.
    
    Returns:
        predicted_accept_length (int)
    """
    candidates = hook_state["draft_candidates"]  # shape: [K+1]

    accept_len = 0
    # candidates[0] 是第一个 speculative token
    for tok in candidates[0]:
        if tok.item() == 0:
            break
        accept_len += 1

    return accept_len - 1


def fully_exit_hook(hook_state):
    """
    Fully exit at each step.
    
    Returns:
        predicted_accept_length (int)
    """
    candidates = hook_state["draft_candidates"]  # shape: [K+1]
    # print(candidates)
    return len(candidates[0]) - 1

# Load your model
model_device = "cuda:4"
baseline_model_device = "cuda:5"

depth = 7
base_model_path = "/data/chenjunyi/models/qwen3-8b"
baseline_EAGLE_model_path = "/data/chenjunyi/models/qwen3-8b-eagle3"
EAGLE_model_path = "/data/chenjunyi/project/SpecForge/outputs/qwen3-8b-eagle3-unkhead_only-dynamic-sharegpt/epoch_9_step_145000"
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=model_device,
    total_token=depth+1,
    depth=depth,
    top_k=1,
)

baseline_model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=baseline_EAGLE_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=baseline_model_device,
    total_token=depth+1,
    depth=depth,
    top_k=1,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
model.eval()

prompt = "Complete the code I provided.from typing import List\n\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\" Filter an input list of strings only for ones that contain given substring\n    >>> filter_by_substring([], 'a')\n    []\n    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n    ['abc', 'bacd', 'array']\n    \"\"\"\n"
messages = [
    {"role": "user", "content": prompt}
]
# Note: this draft model is used for thinking mode disabled
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

input_ids = model_inputs["input_ids"]
prompt_len = input_ids.shape[1]

max_new_tokens = 2048

# Generate with early exit hook and collect statistics
output_ids, stats = model.eagenerate_with_unk_head(
    input_ids=input_ids.to(model_device),
    log=True,  # Important: set log=True to get statistics
    max_new_tokens=max_new_tokens,
    top_k=1,
    early_exit_hook=mtp_sampling_early_exit_hook,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

baseline_output_ids, stats_baseline = baseline_model.eagenerate_with_early_exit_hook(
    input_ids=input_ids.to(baseline_model_device),
    log=True,
    max_new_tokens=max_new_tokens,
    top_k=1,
    early_exit_hook=baseline_sampling_early_exit_hook,  # No early exit
)

print(tokenizer.decode(baseline_output_ids[0], skip_special_tokens=True))

# Compare the average accepted lengths
avg_accepted_length = np.mean(stats['true_accept_lengths'])
avg_accepted_length_baseline = np.mean(stats_baseline['true_accept_lengths'])
print(f"Average accepted length (dynamic EAGLE): {avg_accepted_length}")
print(f"Average accepted length (baseline EAGLE): {avg_accepted_length_baseline}")
print("==========================")

# Compare the average extra verify tokens
extra_verify_tokens = np.array(stats['predicted_accept_lengths']) - np.array(stats['true_accept_lengths'])
extra_verify_tokens_baseline = np.array(stats_baseline['predicted_accept_lengths']) - np.array(stats_baseline['true_accept_lengths'])
avg_extra_verify_tokens = np.mean(extra_verify_tokens)
avg_extra_verify_tokens_baseline = np.mean(extra_verify_tokens_baseline)
print(f"Average extra verify tokens (dynamic EAGLE): {avg_extra_verify_tokens}")
print(f"Average extra verify tokens (baseline EAGLE): {avg_extra_verify_tokens_baseline}")
print("==========================")

# Compare the save numbers of verified tokens
saved_verified_tokens = np.sum(np.array(stats['predicted_accept_lengths'])) - np.sum(np.array(stats_baseline['predicted_accept_lengths']))
avg_saved_verified_tokens = saved_verified_tokens / stats['total_tokens']
print(f"Average saved verified tokens (dynamic EAGLE vs baseline EAGLE): {avg_saved_verified_tokens}")
print("==========================")

