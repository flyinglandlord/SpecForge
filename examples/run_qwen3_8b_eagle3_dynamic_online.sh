#!/bin/bash

# ===================================================================
# Dynamic Training Script for Qwen3-8B EAGLE3
# ===================================================================
# This script trains an EAGLE3 draft model with dynamic stopping mechanism.
# 
# Training Strategy:
# - For each position, the target model generates num_draft_tokens tokens
# - Find the first position where target model's prediction differs from ground truth
# - Replace that position and all subsequent positions with <IDK> token
# - Draft model learns to predict the modified sequence (including <IDK>)
# 
# Benefits:
# - Model learns when to stop drafting (by predicting <IDK>)
# - Reduces wasted computation on low-quality predictions
# - Dynamic draft length instead of fixed num_draft_tokens
# ===================================================================

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# Training configuration
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

# Dynamic training specific settings
TTT_LENGTH=7  # Number of draft tokens to predict
OUTPUT_DIR=$ROOT_DIR/outputs/qwen3-8b-eagle3-dynamic-sharegpt-full-freeze-vocab-top16loss

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /data/chenjunyi/models/qwen3-8b \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3-full-vocab.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --lm-head-key lm_head.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --use-dynamic-length-training \
    --ttt-length $TTT_LENGTH \
    --report-to wandb \
    --wandb-project DEAGLE-train \
    --wandb-name SoftCE_Top16Loss_FullFreezeVocab_sharegpt_dynamic_ttt7 \
    --wandb-entity junyi-chen-sjtu \
    --use-main-model-lm-head