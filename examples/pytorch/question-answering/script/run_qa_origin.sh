set -x
T=`date +%m%d%H%M`

python run_qa.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --pad_to_max_length \
  --doc_stride 128 \
  --output_dir /tmp/test_squad/ \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_all384.$T