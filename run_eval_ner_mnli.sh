#!/bin/sh

python train_bert_ner.py \
  --model_name_or_path output_dir_ner_mnli \
  --dataset_name Babelscape/wikineural \
  --label_column_name ner_tags \
  --output_dir output_dir_ner_mnli/eval_for_ner \
  --do_train False \
  --do_eval True \
  --ignore_mismatched_sizes True