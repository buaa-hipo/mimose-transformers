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

python3 run_swag.py \
  --model_name_or_path roberta-base \
  --output_dir /tmp/test-swag-no-trainer \
  --num_train_epochs 1 \
  --do_train \
  --per_device_train_batch_size 16 \
  --dynamic_checkpoint \
  --warmup_iters 20 \
  --memory_buffer 0.8 \
  --memory_threshold $1 \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_dc$1_roberta.$T
