python run_t5_mlm.py \
	--do_train \
	--do_eval \
	--model_name_or_path t5-small \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
	--output_dir /tmp/t5_wikitext \
	--cache_dir /ssd/huggingface-cache
