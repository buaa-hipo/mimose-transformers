#!/bin/bash

set -x

T=$(date +%m%d%H%M)

BUFFER="${2:-2}"
WARMUP="${3:-10}"

python run_translation.py \
    --model_name_or_path t5-base \
    --do_train \
    --source_lang en_XX \
    --target_lang fr_XX \
    --dataset_name un_pc \
    --dataset_config_name en-fr \
    --output_dir /tmp/t5-base-un \
    --source_prefix 'translate English to French: ' \
    --per_device_train_batch_size=8 \
    --max_train_samples 4000 \
    --max_source_length 512 \
    --max_target_length 512 \
    --num_train_epochs 1 \
    --overwrite_output_dir \
    --cache_dir /ssd/huggingface-cache \
    --predict_with_generate \
    --max_input_size 512 \
    --min_input_size 21 \
    --warmup_iters $WARMUP \
    --dynamic_checkpoint \
    --static_checkpoint \
    --memory_threshold $1 \
    --memory_buffer ${BUFFER} \
    2>&1 | tee log/log.train_log_dc_static$1_${BUFFER}_warmup${WARMUP}_t5.$T
