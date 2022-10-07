set -x

T=`date +%m%d%H%M`
export TASK_NAME=qqp

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir /tmp/$TASK_NAME/ \
  --num_train_epochs 1 \
  --use_sublinear \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_${TASK_NAME}_sublinear.$T