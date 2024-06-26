#!/bin/bash
watermark=$1
pythonenv=${3:-"python3"}
llama=${2:-"meta-llama/Llama-2-7b-chat-hf"}
gpu=${4:-"0"}
device=${5:-"0"}

export CUDA_VISIBLE_DEVICES=$gpu
export TORCH_HOME='/nobackup/users/maxdan/huggingface_cache/'
output_file="/nobackup/users/maxdan/data/sampling-distill-train-data/${watermark}_llama_2_7b_owt_len256_640k.json"
saved_prompts="/nobackup/users/maxdan/data/sampling-distill-train-data/${watermark}_llama_2_7b_prompts_len256_640k.pt"
output_train_file="/nobackup/users/maxdan/data/sampling-distill-train-data/${watermark}_llama_2_7b_owt_len256_640k_train.json"
watermark_config_file="experiments/watermark-configs/${watermark}-config.json"

$pythonenv experiments/generate_sampling_distill_train_data_worker.py \
    --model_name "${llama}" \
    --dataset_name Skylion007/openwebtext \
    --dataset_split train \
    --data_field "text" \
    --streaming \
    --output_file "${output_file}" \
    --output_train_file "${output_train_file}" \
    --num_samples 640000 \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --prompt_length 50 \
    --seed 42 \
    --watermark_config_file "${watermark_config_file}" \
    --save_interval 64000 \
    --fp16 \
    --dataloader_batch_size 10000 \
    --batch_size 128 \
    --device_id $device \
    --total_replicas 12 \
    --saved_prompts "${saved_prompts}" \
#    --input_file "${output_file}" \
#   --dataset_num_skip 630000
