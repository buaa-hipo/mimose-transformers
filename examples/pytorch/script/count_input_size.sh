cd multiple-choice
pwd
for BATCH_SIZE in 1 4 8 16 32 64 128 256
do
    sh count_input_size.sh $BATCH_SIZE
done

cd ../question-answering
pwd
for BATCH_SIZE in 1 4 8 16 32 64 128 256
do
    sh count_input_size.sh $BATCH_SIZE
done

cd ../text-classification
pwd
for TASK_NAME in "cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli"
do
    for BATCH_SIZE in 1 4 8 16 32 64 128 256
    do
        sh count_input_size.sh $TASK_NAME $BATCH_SIZE
    done
done

cd ../token-classification
pwd
for BATCH_SIZE in 1 4 8 16 32 64 128 256
do
    sh count_input_size.sh $BATCH_SIZE
done