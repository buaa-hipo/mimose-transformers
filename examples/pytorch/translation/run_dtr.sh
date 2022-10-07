set -x

T=`date +%m%d%H%M`

MEMORY=${1:-8}

python run_translation.py \
    --model_name_or_path facebook/mbart-large-en-ro  \
    --do_train \
    --do_eval \
    --num_train_epochs 1 \
    --use_dtr \
    --memory_threshold ${MEMORY} \
    --dataset_name wmt16 \
    --dataset_config_name ro-en \
    --source_lang en_XX \
    --target_lang ro_RO \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate 2>&1 | tee log/log.train_log_dtr_${MEMORY}.$T