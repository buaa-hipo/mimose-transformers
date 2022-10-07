set -x
T=`date '+%Y-%m-%d-%H-%M-%S'`

python run_qa_beam_search.py \
  --model_name_or_path xlnet-base-cased \
  --dataset_name squad \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 1 \
  --doc_stride 128 \
  --output_dir /tmp/test_squad/ \
  --max_seq_length 512 \
  --memory_threshold $1 \
  --overwrite_output_dir 2>&1 | tee train_log/log.train_xlnet_reserved_mem.$T
