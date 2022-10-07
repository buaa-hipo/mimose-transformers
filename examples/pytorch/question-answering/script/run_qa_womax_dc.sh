#!/bin/bash

set -x
T=`date +%m%d%H%M`

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --output_dir /tmp/test_squad/ \
  --dynamic_checkpoint \
  --warmup_iters 30 \
  --memory_threshold $1 \
  --max_input_size 516 \
  --min_input_size 146 \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_dc_$1.$T