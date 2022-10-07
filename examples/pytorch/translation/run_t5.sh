set -x

T=$(date +%m%d%H%M)

python run_translation.py \
    --model_name_or_path t5-large \
    --do_train \
    --source_lang en_XX \
    --target_lang de_XX \
    --dataset_name para_crawl \
    --dataset_config_name ende \
    --output_dir /tmp/tst-translation \
    --source_prefix 'translate English to German: ' \
    --per_device_train_batch_size=4 \
    --num_train_epochs 1 \
    --overwrite_output_dir \
	--cache_dir /ssd/huggingface-cache
    #--predict_with_generate 2>&1 | tee log/log.train_log.$T
