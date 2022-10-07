python run_mlm.py \
	--model_name_or_path bert-base-uncased \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--do_train \
	--do_eval \
	--dataset_name 'allenai/c4' \
	--dataset_config_name en \
	--data_files 'en/c4-train.00001-of-01024.json.gz' \
	--cache_dir /ssd/huggingface-cache \
	--output_dir /tmp/test-mlm

# --dataset_name wikitext \
# --dataset_config_name wikitext-2-raw-v1 \
