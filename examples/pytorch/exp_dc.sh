#!/bin/bash

set -x

# multiple-choice
cd multiple-choice
rm ~/.cache/huggingface/datasets/swag/ -r 
./run_swag_roberta_dc.sh 4
sleep 300
./run_swag_roberta_dc.sh 5
sleep 300
./run_swag_roberta_dc.sh 6
sleep 300
./run_swag_roberta_dc.sh 7
sleep 300
./run_swag_roberta_dc.sh 8
sleep 300
cp -f train_log/* ../LOG-PROCESS/mc_dataset/.

# question-answering
cd ../question-answering
./run_qa_womax_dc.sh 4
sleep 300
./run_qa_womax_dc.sh 5
sleep 300
./run_qa_womax_dc.sh 6
sleep 300
./run_qa_womax_dc.sh 7
sleep 300
./run_qa_womax_dc.sh 8
sleep 300
./run_qa_xlnet_dc.sh 4
sleep 300
./run_qa_xlnet_dc.sh 5
sleep 300
./run_qa_xlnet_dc.sh 6
sleep 300
./run_qa_xlnet_dc.sh 7
sleep 300
./run_qa_xlnet_dc.sh 8
sleep 300
cp -f train_log/* ../LOG-PROCESS/qa_dataset/.

# text-classification
cd ../text-classification
./run_dc.sh 4
sleep 300
./run_dc.sh 5
sleep 300
./run_dc.sh 6
sleep 300
./run_dc.sh 7
sleep 300
./run_dc.sh 8
sleep 300
cp -f train_log/* ../LOG-PROCESS/tc_dataset/.

