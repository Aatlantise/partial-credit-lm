#!/bin/bash

# Array of your checkpoint directories
MODELS=(
    "../results/standard/checkpoint-1140"
    "../results/partial_eps0.1/checkpoint-1140"
    "../results/partial_eps0.5/checkpoint-1140"
    "../results/partial_eps0.9/checkpoint-1140"
    "../results/class_only/checkpoint-1140"
)

for MODEL_PATH in "${MODELS[@]}"; do
    # Extract the model name for the output file
    PARENT_DIR="${MODEL_PATH%/*}"

    # 2. Extract just the name: standard
    MODEL_NAME=$(basename "$PARENT_DIR")
    echo "========================================"
    echo "Evaluating $MODEL_NAME on BLiMP..."
    echo "========================================"

    lm_eval \
        --model hf \
        --model_args pretrained="$MODEL_PATH",add_bos_token=True,tokenizer=gpt2 \
        --tasks blimp \
        --device cuda:0 \
        --batch_size 64 \
        --output_path "./eval_results/${MODEL_NAME}_blimp.json"
done

echo "All evaluations complete!"