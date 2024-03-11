#!/bin/sh

export TASK_NAME=mnli

python train_bert_glue.py \
  --model_name_or_path output_dir_mnli_ner \
  --task_name $TASK_NAME \
  --do_train False \
  --do_eval True \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --output_dir output_dir_mnli_ner/eval_for_mnli \
  --ignore_mismatched_sizes True