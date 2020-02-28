## BERT_CLS

### 基本概念

* 多类分类`Multiclass classification`，即类别之间互斥，`二分类`是其特例
* 多标签分类`Multilabal classification`，即类别之间不互斥

### 支持任务

* 二分类+输出分类/回归结果
* 多类分类+输出分类结果
* 多标签分类+输出分类结果

### scripts

* `multiclass&classification train&eval`

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  export MAX_LENGTH=256
  export BERT_MODEL=bert-base-chinese
  export BATCH_SIZE=16
  export NUM_EPOCHS=3
  export SAVE_STEPS=750
  export SEED=0
  export DATA_DIR=PATH_TO_DATA
  
  python3 run_bert_cls.py --data_dir $DATA_DIR \
  --model_type bert \
  --output_mode classification \
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
  --overwrite_output_dir \
  --train_data_number 10000 \
  --logging_steps 50
  ```

* `multilabel&classification train&eval`

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  export MAX_LENGTH=256
  export BERT_MODEL=bert-base-chinese
  export BATCH_SIZE=16
  export NUM_EPOCHS=3
  export SAVE_STEPS=750
  export SEED=0
  export DATA_DIR=PATH_TO_DATA
  
  python3 run_bert_cls.py --data_dir $DATA_DIR \
  --model_type bert \
  --output_mode classification \
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
  --overwrite_output_dir \
  --multilabel \
  --train_data_number 10000 \
  --logging_steps 50
  ```


### 数据准备

* 格式

  ```python
  ## data.json
  [
    {
      "index": int,
      "text_a": str,
      "text_b": str,  # default: None
      # 多类分类 or 多标签分类
      "label": str or [str]  # or None for test data
    }, ...
  ]
  ```

* 示例

  ```python
  ## data.json
  {
    "index": "00091edb8f43841ef781cb7cd36b1485",
    "text_a": "尾灯同样采用的是LED光源，与前脸呼应， 确保2019款BMW 3系在每个角度看去都运动不凡。",
    "label": ['尾灯']  # 多标签分类
  }
  
  ## labels.json
  ["天窗", "尾灯", "大灯", ...] 
  ```
  

### 测试实验结果

`export DATA_DIR=/data/home/liusunan/eigen/liusunan_projects/car_part/experiments/0227/multilabel_dat`