#!/bin/bash
watermark=$1
train_file=$2
out_dir=$3
port=$4
alpaca=${5:-"wxjiao/alpaca-7b"}

#train_file="data/sampling-distill-train-data/${watermark}_llama_2_7b_owt_len256_640k_train.json"
watermark_config_file="experiments/watermark-configs/${watermark}-config.json"
model_name="alpaca-7b-sampling-watermark-distill-${watermark}"

if [[ "$watermark" == kth* ]]; then
    group_texts="False"
else
    group_texts="True"
fi

torchrun --nproc_per_node=4 --master_port=${port} train_sampling_distill.py \
    --model_name_or_path "${alpaca}" \
    --train_file "${train_file}" \
    --watermark_config_file "${watermark_config_file}" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size 1024 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --num_train_epochs 3 \
    --group_texts ${group_texts} \
    --tf32 True \
    --bf16 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \

