#!/bin/bash

set -x

# multiple-choice
cd multiple-choice
rm ~/.cache/huggingface/datasets/swag/ -r 
./run_swag_roberta_sublinear.sh
sleep 300
cp -f train_log/* ../LOG-PROCESS/mc_dataset/.

# question-answering
cd ../question-answering
./run_qa_womax_sublinear.sh
sleep 300
./run_qa_xlnet_sublinear.sh
sleep 300
cp -f train_log/* ../LOG-PROCESS/qa_dataset/.

# text-classification
cd ../text-classification
./run_sublinear.sh
cp -f train_log/* ../LOG-PROCESS/tc_dataset/.

