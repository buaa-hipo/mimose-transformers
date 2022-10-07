set -x

T=`date '+%Y-%m-%d-%H-%M-%S'`
export TASK_NAME=qqp

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --per_device_train_batch_size 32 \
  --num_train_epochs 1 \
  --learning_rate 2e-5 \
  --output_dir /tmp/$TASK_NAME/ \
  --num_train_epochs 1 \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_$TASK_NAME.$T