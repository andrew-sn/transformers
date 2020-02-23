export CUDA_VISIBLE_DEVICES=0
export MAX_LENGTH=128
export BERT_MODEL=bert-base-chinese
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
export DATA_DIR=PATH_TO_DATA


export CUDA_VISIBLE_DEVICES=0
export MAX_LENGTH=128
export BERT_MODEL=bert-base-chinese
export BATCH_SIZE=8
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1
export DATA_DIR=PATH_TO_DATA

#`crf&init_transitions&multi_decode train&eval
python3 run_bert_crf.py --data_dir $DATA_DIR \
--model_type bert \
--labels $DATA_DIR/labels.json \
--model_name_or_path $BERT_MODEL \
--output_dir $DATA_DIR/ckpt \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--crf \
--init_transitions \
--crf_decode_topk=2