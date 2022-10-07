#!/bin/bash

set -x
T=`date '+%Y-%m-%d-%H-%M-%S'`

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --output_dir /tmp/test_squad/ \
  --gradient_checkpointing \
  --memory_threshold $1 \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_womax_sublinear_reserved_mem.$T