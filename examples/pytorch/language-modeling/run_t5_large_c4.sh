#!/bin/bash

python run_t5_mlm.py \
	--do_train \
	--model_name_or_path t5-large \
	--per_device_train_batch_size 6 \
	--dataset_name 'allenai/c4' \
	--dataset_config_name en \
	--data_files 'en/c4-train.00001-of-01024.json.gz' \
	--output_dir /tmp/t5_c4/large/output \
	--cache_dir /ssd/huggingface-cache
	# --gradient_checkpointing \
	# --per_device_train_batch_size 16 \
