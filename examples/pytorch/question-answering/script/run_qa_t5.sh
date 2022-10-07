set -x
T=`date +%m%d%H%M`


python run_seq2seq_qa.py \
  --model_name_or_path t5-small \
  --dataset_name squad_v2 \
  --context_column context \
  --question_column question \
  --answer_column answers \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --max_seq_length 384 \
  --output_dir /tmp/test_squad/ \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_t5.$T