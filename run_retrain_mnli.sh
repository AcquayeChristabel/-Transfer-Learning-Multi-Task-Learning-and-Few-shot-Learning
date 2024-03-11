#!/bin/sh

python train_bert_ner.py \
  --model_name_or_path output_dir_mnli \
  --dataset_name Babelscape/wikineural \
  --label_column_name ner_tags \
  --output_dir output_dir_mnli_ner \
  --do_train True \
  --do_eval True \
  --num_train_epochs 3 \
  --save_total_limit 3 \
  --ignore_mismatched_sizes True