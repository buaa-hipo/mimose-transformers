set -x

T=`date +%m%d%H%M`
TASK_NAME=${1:-'qqp'}
BATCH_SIZE=${2:-'32'}

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate 2e-5 \
  --output_dir /tmp/$TASK_NAME/ \
  --num_train_epochs 1 \
  --only_input_size \
  --overwrite_output_dir 2>&1 | tee train_log/log.dataset_glue_${TASK_NAME}_${BATCH_SIZE}