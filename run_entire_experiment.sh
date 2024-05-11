OUTPUT_DIR=/home/eliotcow/spoofoff/saved_models/
TORCH_RUN_PORT=29500
PYTHONENVPATH=/nobackup/users/maxdan/anaconda3/bin/python
PATH_TO_OUTPUT = /home/eliotcow/spoofoff/train_data

# Create training data
bash scripts/train/generate_sampling_distill_train_data.sh kgw-k1-gamma0.25-delta2 meta-llama/Llama-2-7b-chat-hf $PYTHONENVPATH
# Filter out refusals
bash scripts/train/filter_distill_train_data.sh $PATH_TO_TRAINING $PATH_TO_OUTPUT $PYTHONENVPATH
# Train model
bash scripts/train/train_alpaca_sampling_distill.sh kgw-k1-gamma0.25-delta2 $OUTPUT_DIR $TORCH_RUN_PORT wxjiao/alpaca-7b
# Evaluate!


