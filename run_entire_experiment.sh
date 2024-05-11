OUTPUT_DIR=/home/eliotcow/spoofoff/saved_models/
TORCH_RUN_PORT=29500
# Create training data
bash scripts/train/generate_sampling_distill_train_data.sh kgw-k1-gamma0.25-delta2 meta-llama/Llama-2-7b-chat-hf
# Filter out refusals
# TODO: Implement this
# Train model
bash scripts/train/train_alpaca_sampling_distill.sh kgw-k1-gamma0.25-delta2 $OUTPUT_DIR $TORCH_RUN_PORT wxjiao/alpaca-7b
# Evaluate!


