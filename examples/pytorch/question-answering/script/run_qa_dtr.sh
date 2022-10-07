set -x
T=`date +%m%d%H%M`

MEMORY=${1:-"6"}

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --use_dtr \
  --memory_threshold $MEMORY \
  --output_dir /tmp/test_squad/ \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_dtr_${MEMORY}.$T