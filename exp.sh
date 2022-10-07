#!/bin/bash

set -x

export PROJECT_ROOT=$(pwd)
export CUDA_VISIBLE_DEVICES=0

# TR-T5
cd $PROJECT_ROOT/examples/pytorch/translation
mkdir -p train_log && \
for i in 7 9 13 14; do
	bash run_t5_base_un_4000_dc.sh $i
done

# QA-Bert
cd $PROJECT_ROOT/examples/pytorch/question-answering
mkdir -p train_log && \
for i in 4 5 6 7 8; do
	bash run_qa_womax_dc.sh $i
done

# MC-Roberta
cd $PROJECT_ROOT/examples/pytorch/multiple-choice
mkdir -p train_log && \
for i in 4 5 6 7 8; do
	bash run_swag_roberta_dc.sh $i
done

# TC-Bert
cd $PROJECT_ROOT/examples/pytorch/text-classification
mkdir -p train_log && \
for i in 4 5 6 7 8; do
	bash run_dc.sh $i
done