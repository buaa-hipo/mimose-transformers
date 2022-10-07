set -x
T=`date +%m%d%H%M`

MODEL="bert-base-uncased"
EPOCHS=3
LOGGING_STEPS=100

# # regular
# python run_qa.py \
#   --model_name_or_path $MODEL \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs $EPOCHS \
#   --doc_stride 128 \
#   --output_dir /tmp/test_squad/ \
#   --logging_steps ${LOGGING_STEPS} \
#   --overwrite_output_dir 2>&1 | tee train_log/log.loss_regular.$T


# DC
python run_qa.py \
  --model_name_or_path $MODEL \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs $EPOCHS \
  --doc_stride 128 \
  --output_dir /tmp/test_squad/ \
  --dynamic_checkpoint \
  --warmup_iters 30 \
  --memory_threshold 6 \
  --max_input_size 516 \
  --min_input_size 146 \
  --logging_steps ${LOGGING_STEPS} \
  --overwrite_output_dir 2>&1 | tee train_log/log.loss_dc_6.$T


# # Sublinear
# python run_qa.py \
#   --model_name_or_path $MODEL \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs $EPOCHS \
#   --doc_stride 128 \
#   --output_dir /tmp/test_squad/ \
#   --use_sublinear \
#   --logging_steps ${LOGGING_STEPS} \
#   --overwrite_output_dir 2>&1 | tee train_log/log.loss_sublinear.$T

# # GC
# python run_qa.py \
#   --model_name_or_path $MODEL \
#   --dataset_name squad \
#   --do_train \
#   --do_eval \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs $EPOCHS \
#   --doc_stride 128 \
#   --output_dir /tmp/test_squad/ \
#   --gradient_checkpointing \
#   --logging_steps ${LOGGING_STEPS} \
#   --overwrite_output_dir 2>&1 | tee train_log/log.loss_gc.$T
