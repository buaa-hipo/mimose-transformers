set -x
T=`date +%m%d%H%M`

BATCH_SIZE=${1:-'12'}

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --output_dir /tmp/test_squad/ \
  --only_input_size \
  --overwrite_output_dir 2>&1 | tee train_log/log.dataset_squad_${BATCH_SIZE}