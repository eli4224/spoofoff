#!/bin/bash
dataset=${1:-"harmfulqa"}
output_file=$2
llama=$3
ppl_model=$4
pythonenv=${5:-"python3"}

shift 5
models="$@"
tokenizer=$1

# scripts/evaluate/generate_and_evaluate.sh harmfulqa /nobackup/users/maxdan/generate_and_evaluate_1/out.txt meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-7b-chat-hf /nobackup/users/maxdan/anaconda3/bin/python meta-llama/Llama-2-7b-chat-hf
shift 5
models="$@"
tokenizer=$1

num_tokens=200
prompt_length=31
num_samples=1960

if [ "$dataset" = "c4" ]; then
    dataset_args="--dataset_name allenai/c4 \
    --dataset_config_name realnewslike \
    --dataset_split validation \
    --data_field text"
elif [ "$dataset" = "wikipedia" ]; then
    dataset_args="--dataset_name wikipedia \
    --dataset_config_name 20220301.en \
    --dataset_split train \
    --data_field text"
elif [ "$dataset" = "arxiv" ]; then
    dataset_args="--dataset_name scientific_papers \
    --dataset_config_name arxiv \
    --dataset_split test \
    --data_field article"
elif [ "$dataset" = "harmfulqa" ]; then
    dataset_args="--dataset_name declare-lab/HarmfulQA \
    --dataset_config_name harmfulqa \
    --dataset_split train \
    --data_field question"
else
    echo "Unsupported dataset ${dataset}."
    exit 1
fi

$pythonenv experiments/generate_samples.py \
    --model_names ${models} \
    ${dataset_args} \
    --overwrite_output_file \
    --streaming \
    --fp16 \
    --output_file "${output_file}" \
    --num_samples ${num_samples} \
    --min_new_tokens ${num_tokens} \
    --max_new_tokens ${num_tokens} \
    --prompt_length ${prompt_length} \
    --batch_size 1 \
    --seed 42 \
    --watermark_config_filename experiments/watermark-configs/kgw-k1-gamma0.25-delta2-config.json

$pythonenv experiments/compute_metrics.py \
    --input_file "${output_file}"  \
    --output_file "${output_file}" \
    --overwrite_output_file \
    --tokenizer_name "${tokenizer}" \
    --watermark_tokenizer_name "${llama}" \
    --truncate \
    --num_tokens ${num_tokens} \
    --ppl_model_name "${ppl_model}" \
    --fp16 \
    --batch_size 16 \
    --metrics p_value rep ppl

# KTH watermark detection takes a while (several hours) and only requires CPU,
# you can comment this out and run separately if desired
$pythonenv watermarks/kth/compute_kth_scores.py \
    --tokenizer_name "${llama}" \
    --input_file "${output_file}" \
    --output_file "${output_file}" \
    --num_samples ${num_samples} \
    --num_tokens ${num_tokens} \
    --gamma 0.0 \
    --ref_dist_file "data/${dataset}/kth_ref_distribution_llama_${dataset}.json" \

$pythonenv experiments/compute_auroc.py \
    --input_file "${output_file}" \
    --output_file "${output_file}" \
    --overwrite_output_file \
    --auroc_ref_dist_file "data/${dataset}/auroc_ref_distribution_llama_${dataset}.json" \
    --kth_ref_dist_file "data/${dataset}/kth_ref_distribution_llama_${dataset}.json"
