set -x

T=`date +%m%d%H%M`
export TASK_NAME=qqp

MODEL="bert-base-cased"
EPOCHS=3
LOGGING_STEPS=100

# # regular
# python run_glue.py \
#   --model_name_or_path $MODEL \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --output_dir /tmp/$TASK_NAME/ \
#   --num_train_epochs $EPOCHS \
#   --logging_steps ${LOGGING_STEPS} \
#   --overwrite_output_dir 2>&1 | tee train_log/log.loss_${TASK_NAME}_regular.$T

# DC
python run_glue.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir /tmp/$TASK_NAME/ \
  --num_train_epochs $EPOCHS \
  --logging_steps ${LOGGING_STEPS} \
  --dynamic_checkpoint \
  --warmup_iters 30 \
  --memory_threshold 6 \
  --max_input_size 332 \
  --min_input_size 30 \
  --overwrite_output_dir 2>&1 | tee train_log/log.loss_${TASK_NAME}_dc_6.$T

# # sublinear
# python run_glue.py \
#   --model_name_or_path $MODEL \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --output_dir /tmp/$TASK_NAME/ \
#   --num_train_epochs $EPOCHS \
#   --logging_steps ${LOGGING_STEPS} \
#   --use_sublinear \
#   --overwrite_output_dir 2>&1 | tee train_log/log.loss_${TASK_NAME}_sublinear.$T


# # GC
# python run_glue.py \
#   --model_name_or_path $MODEL \
#   --task_name $TASK_NAME \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 32 \
#   --learning_rate 2e-5 \
#   --output_dir /tmp/$TASK_NAME/ \
#   --num_train_epochs $EPOCHS \
#   --logging_steps ${LOGGING_STEPS} \
#   --gradient_checkpointing \
#   --overwrite_output_dir 2>&1 | tee train_log/log.loss_${TASK_NAME}_gc.$T


