## BERT_CLS_WITH_CAT(MULTITASK)
> 带类别Cat信息的multi-class句子分类任务

### 最佳实践

* 首先考虑是否要利用这部分Cat信息
* 如果要使用，首先考虑非`multitask`模型
* 如果要使用`multitask`，首先考虑主次任务之间是否存在关联
* 如果仍然要使用`multitask`，建议开始`--step`模式，即将次任务的学习成果融入主任务

### 模式说明

* `non-multitask模式`: 使用词向量表达Cat信息并简单拼接
* `multitask模式`: 将主次任务的loss按比例相加或使用`unweighted`模式相加
* `multitask&step模式`: 在`multitask`模式的基础上，将次任务的学习结果融入主任务

### scripts

* `non-multitask模式`

  ```shell
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
  --labels $DATA_DIR/labels.json \  # 主任务标签
  --categories $DATA_DIR/categories.json \  # 次任务标签
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
  --category_hidden_size 100 \
  --learning_rate 5e-5 \
  --category_hidden_size 10  # 次任务标签向量长度
  ```

* `multitask模式`

  ```shell
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
  --labels $DATA_DIR/labels.json \  # 主任务标签
  --categories $DATA_DIR/categories.json \  # 次任务标签
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
  --category_hidden_size 100 \
  --learning_rate 5e-5 \
  --multitask \
  --main_task_ratio 0.5  # 手动设置主次任务的loss比例
  ```

* `multitask&step模式`

  ```shell
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
  --labels $DATA_DIR/labels.json \  # 主任务标签
  --categories $DATA_DIR/categories.json \  # 次任务标签
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
  --category_hidden_size 100 \
--learning_rate 5e-5 \
  --multitask \
  --step
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
      "category": str,
      "label": str  # or None for test data
    }, ...
  ]
  ```

* 示例

  ```python
  ## data.json
  {
    "index": "00091edb8f43841ef781cb7cd36b1485",
    "text_a": "尾灯同样采用的是LED光源，与前脸呼应， 确保2019款BMW 3系在每个角度看去都运动不凡。",
    "category": "描述"
    "label": "尾灯"
  }
  
  ## labels.json 普通 or 二分类
  ["天窗", "尾灯", "大灯", ...] or [0, 1]
  ```
  

### 测试数据概况

|       | 数据量 | 最大长度 |
| ----- | ------ | -------- |
| train | 8747   | 40       |
| dev   | 2002   | 38       |

类别: `{'哮喘', '胸膜炎', '咳血', '上呼吸道感染', '感冒', '肺气肿', '肺炎', '支原体肺炎'}`

### 测试实验结果

| 实验编号 |   F1_micro    |   F1_macro    | SUB_F1_micro  | SUB_F1_macro  |            主要参数            |                           收敛情况                           |            备注             |
| :------: | :-----------: | :-----------: | :-----------: | :-----------: | :----------------------------: | :----------------------------------------------------------: | :-------------------------: |
|    1     | 0.9406(1.4k)  | 0.9383(1.4k)  |       -       |       -       |            Batch16             |                              -                               |            BERT             |
|    2     | 0.9306(1.45k) | 0.9275(1.45k) |       -       |       -       |   Batch16&lr5e-5&catHidden10   |                         loss 微震荡                          |        BERT+cat编码         |
|    3     | 0.9341(1.6k)  | 0.9314(1.6k)  |       -       |       -       |   Batch16&lr5e-6&catHidden10   |                              -                               |        BERT+cat编码         |
|    4     | 0.9341(1.6k)  | 0.9314(1.6k)  |       -       |       -       |  Batch16&lr5e-6&catHidden100   |                              -                               | BERT+cat编码 与catH10无区别 |
|    5     | 0.9146(3.2k)  | 0.9111(3.2k)  |  0.946(3.2k)  |  0.944(3.2k)  |   Batch8&lr5e-5&mainloss0.5    | main_loss震荡，sub_loss较平滑，整体loss平滑 main比sub在loss上大约是2:1 |       BERT+multitask        |
|    6     | 0.9146(3.25k) | 0.9113(3.25k) | 0.944(3.25k)  | 0.939(3.25k)  |       Batch8&unweighted        |         main_loss震荡，sub_loss较平滑，整体loss平滑          |  BERT+multitask+unweighted  |
|    7     | 0.9181(3.25k) | 0.9148(3.25k) | 0.9416(3.25k) | 0.9342(3.25k) |   Batch8&lr5e-5&mainloss0.66   |         main_loss震荡，sub_loss较平滑，整体loss平滑          |       BERT+multitask        |
|    8     | 0.9071(3.25k) | 0.9033(3.25k) | 0.9391(3.25k) | 0.9281(3.25k) |    Batch8&lr5e-6&unweighted    |         main_loss震荡，sub_loss较平滑，整体loss平滑          |  BERT+multitask+unweighted  |
|    9     | 0.9216(4.3k)  | 0.9188(4.3k)  | 0.9491(4.3k)  | 0.9414(4.3k)  | Batch6&lr5e-5&step&mainloss0.5 |         main_loss震荡，sub_loss微震荡，整体loss平滑          |     BERT+multitask+step     |

* 结论
  - 开启`--step`对`multitask`任务帮助最大
  - 在`multitask`任务中使用`unweighted`能显著平衡主次任务的`loss`，但对该任务的帮助不大
  - 使用`multitask`对主任务无帮助

