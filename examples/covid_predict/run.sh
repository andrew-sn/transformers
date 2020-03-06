cd /transformers/examples/covid_predict

python transform.py

export MAX_LENGTH=64
export BERT_MODEL=bert-base-chinese
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=0
export DATA_DIR=./

python RUN.py --data_dir $DATA_DIR \
--model_type bert \
--output_mode classification \
--labels labels.json \
--model_name_or_path $BERT_MODEL \
--output_dir ckpt/checkpoint-1500 \
--max_seq_length $MAX_LENGTH \
--seed $SEED \
--do_eval