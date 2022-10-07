#!/bin/bash

set -x

# if [ $? -ne 0 ]; then
#     echo "failed"
# else


# multiple-choice
cd multiple-choice
rm ~/.cache/huggingface/datasets/swag/ -r 
./run_swag_roberta_reserved_mem.sh 9.95
sleep 300
# rm ~/.cache/huggingface/datasets/swag/ -r
# ./run_swag_roberta_sublinear_reserved_mem.sh 3.47
# sleep 300
cp -f train_log/* ../LOG-PROCESS/mc_dataset/.

# question-answering
cd ../question-answering
# ./run_qa_womax_reserved_mem.sh 9.53
# sleep 300
# ./run_qa_womax_sublinear_reserved_mem.sh 3.1
# sleep 300
./run_qa_xlnet_reserved_mem.sh 14.9
sleep 300
./run_qa_xlnet_sublinear_reserved_mem.sh 4.6
sleep 300
cp -f train_log/* ../LOG-PROCESS/qa_dataset/.

# text-classification
cd ../text-classification
./run_reserved_mem.sh 13.2
# sleep 300
# ./run_sublinear_reserved_mem.sh 3.6
cp -f train_log/* ../LOG-PROCESS/tc_dataset/.

