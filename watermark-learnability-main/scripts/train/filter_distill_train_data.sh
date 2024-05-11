#!/bin/bash
pythonenv=${3:-"python3"}
datapath=${1}
outpath=${2}

$pythonenv experiments/filter_distill_train_data.py \
    --data_path "${datapath}" \
    --out_path "${outpath}" \
