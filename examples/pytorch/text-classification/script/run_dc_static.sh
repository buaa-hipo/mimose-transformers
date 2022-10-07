set -x

T=`date +%m%d%H%M`
export TASK_NAME=qqp

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir /tmp/$TASK_NAME/ \
  --num_train_epochs 1 \
  --dynamic_checkpoint \
  --static_checkpoint \
  --warmup_iters 30 \
  --memory_threshold $1 \
  --max_input_size 332 \
  --min_input_size 30 \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_${TASK_NAME}_dc_static_$1.$T