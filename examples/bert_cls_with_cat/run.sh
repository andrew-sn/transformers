export CUDA_VISIBLE_DEVICES=0
export MAX_LENGTH=64
export BERT_MODEL=bert-base-chinese
export BATCH_SIZE=6
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=0
export DATA_DIR=PATH_TO_DATA

python3 run_bert_cls_with_cat.py --data_dir $DATA_DIR \
--model_type bert \
--labels $DATA_DIR/labels.json \
--categories $DATA_DIR/categories.json \
--model_name_or_path $BERT_MODEL \
--output_dir $DATA_DIR/ckpt \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--overwrite_output_dir \
--logging_steps 100 \
--evaluate_during_training \
--learning_rate 5e-5 \
--multitask \
--main_task_ratio 0.5
