python run_t5_mlm.py \
	--do_train \
	--model_name_or_path t5-base \
	--per_device_train_batch_size 8 \
	--dataset_name 'allenai/c4' \
	--dataset_config_name en \
	--data_files 'en/c4-train.00001-of-01024.json.gz' \
	--output_dir /tmp/t5_c4/base/output \
	--cache_dir /ssd/huggingface-cache
