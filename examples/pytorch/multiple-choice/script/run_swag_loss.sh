# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
T=`date +%m%d%H%M`

EPOCHS=3
MODEL="bert-base-uncased"
LOGGING_STEPS=100

python3 run_swag.py \
  --model_name_or_path ${MODEL} \
  --output_dir /tmp/test-swag-no-trainer \
  --num_train_epochs ${EPOCHS} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --logging_steps ${LOGGING_STEPS} \
  --overwrite_output_dir 2>&1 | tee train_log/log.loss_regular.$T


python3 run_swag.py \
  --model_name_or_path ${MODEL} \
  --output_dir /tmp/test-swag-no-trainer \
  --do_train \
  --do_eval \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size 16 \
  --dynamic_checkpoint \
  --warmup_iters 30 \
  --memory_threshold 6 \
  --max_input_size 142 \
  --min_input_size 40 \
  --logging_steps ${LOGGING_STEPS} \
  --overwrite_output_dir 2>&1 | tee train_log/log.loss_dc_6.$T


python3 run_swag.py \
  --model_name_or_path ${MODEL} \
  --output_dir /tmp/test-swag-no-trainer \
  --do_train \
  --do_eval \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size 16 \
  --use_sublinear \
  --logging_steps ${LOGGING_STEPS} \
  --overwrite_output_dir 2>&1 | tee train_log/log.loss_sublinear.$T


python3 run_swag.py \
  --model_name_or_path ${MODEL} \
  --output_dir /tmp/test-swag-no-trainer \
  --do_train \
  --do_eval \
  --num_train_epochs ${EPOCHS} \
  --per_device_train_batch_size 16 \
  --gradient_checkpointing \
  --logging_steps ${LOGGING_STEPS} \
  --overwrite_output_dir 2>&1 | tee train_log/log.loss_gc.$T

