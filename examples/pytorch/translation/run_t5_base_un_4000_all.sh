#!/bin/bash

set -x

CUDA_VISIBLE_DEVICES=1 bash run_t5_base_un_4000_dc.sh 7 2
CUDA_VISIBLE_DEVICES=0 bash run_t5_base_un_4000_dc.sh 10 2
CUDA_VISIBLE_DEVICES=1 bash run_t5_base_un_4000_dc.sh 13 2
CUDA_VISIBLE_DEVICES=0 bash run_t5_base_un_4000_dc.sh 16 4
CUDA_VISIBLE_DEVICES=1 bash run_t5_base_un_4000_dc_static.sh 7 2
CUDA_VISIBLE_DEVICES=0 bash run_t5_base_un_4000_dc_static.sh 10 2
CUDA_VISIBLE_DEVICES=1 bash run_t5_base_un_4000_dc_static.sh 13 2
CUDA_VISIBLE_DEVICES=0 bash run_t5_base_un_4000_dc_static.sh 16 4
CUDA_VISIBLE_DEVICES=1 bash run_t5_base_un_4000_gradient_checkpoint.sh 6
CUDA_VISIBLE_DEVICES=0 bash run_t5_base_un_4000_baseline.sh 20