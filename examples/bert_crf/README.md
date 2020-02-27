## BERT_CRF

### 最佳实践

* 开启`CRF`会降低GPU的使用效率，因此在大数据集上不建议开启；
* 通过设置`crf_decode_topk > 1`可以开启`crf_multi_decode`，在`更重视Recall`或`标注数据漏标严重`的任务场景中可以使用；

### scripts

* `no_crf train&eval`

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  export MAX_LENGTH=128
  export BERT_MODEL=bert-base-chinese
  export BATCH_SIZE=8
  export NUM_EPOCHS=3
  export SAVE_STEPS=750
  export SEED=1
  export DATA_DIR=PATH_TO_DATA
  
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
  --do_eval
  ```

* `crf&init_transitions train&eval`

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  export MAX_LENGTH=128
  export BERT_MODEL=bert-base-chinese
  export BATCH_SIZE=8
  export NUM_EPOCHS=3
  export SAVE_STEPS=750
  export SEED=1
  export DATA_DIR=PATH_TO_DATA
  
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
  --init_transitions
  ```

* `crf&init_transitions&multi_decode train&eval`

  ```shell
  export CUDA_VISIBLE_DEVICES=0
  export MAX_LENGTH=128
  export BERT_MODEL=bert-base-chinese
  export BATCH_SIZE=8
  export NUM_EPOCHS=3
  export SAVE_STEPS=750
  export SEED=1
  export DATA_DIR=PATH_TO_DATA
  
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
  ```

### 数据准备

* 格式

  ```python
  ## data.json
  [
    {
      "index": int,
      "words": [str],
      "labels": [str]  # or None for test data
    }, ...
  ]
  
  ## labels.json
  [str]
  ```

* 示例

  ```python
  ## data.json
  {
    "index": 2,
    # assert len(words) == len(labels)
    "words": ["外", "观", "：", "中", "庸", "里", "带", "着", "法", "式", "风", "情"],
    "labels": ["O", "O", "O", "B-风格", "I-风格", "O", "O", "O", "O", "O", "O", "O"]
  }
  
  ## labels.json
  ['风格', '专业风格']
  ```

* 注意：

  - 确保`len(words) == len(labels)`
  - 无需处理`[CLS] / [SEP]`等标签
  - 无需处理`空格`，代码会自动将`空格`替换为`[UNK]`

### 测试实验结果

`export DATA_DIR=/data/home/liusunan/eigen/liusunan_projects/car_style/experiments/0222`

* `no_crf`

  ```shell
  2020-02-23 15:29:27,500 : INFO : ***** Eval results  *****
  2020-02-23 15:29:27,500 : INFO :   f1 = 0.5014749262536873
  2020-02-23 15:29:27,500 : INFO :   loss = 0.07045003284628575
  2020-02-23 15:29:27,500 : INFO :   precision = 0.4228855721393035
  2020-02-23 15:29:27,500 : INFO :   recall = 0.6159420289855072
  ```

* `crf&init_transitions`

  ```shell
  2020-02-27 11:09:22,737 : INFO : ***** Eval results  *****
  2020-02-27 11:09:22,737 : INFO :   f1 = 0.56957928802589
  2020-02-27 11:09:22,737 : INFO :   loss = 3.036368516775278
  2020-02-27 11:09:22,737 : INFO :   precision = 0.5146198830409356
  2020-02-27 11:09:22,737 : INFO :   recall = 0.6376811594202898
  2020-02-27 11:09:22,737 : INFO : ***** Eval path score(s) in CRF *****
  2020-02-27 11:09:22,737 : INFO :   Eval No.0 path score = 547.3997802734375
  2020-02-27 11:09:22,737 : INFO :   Eval No.1 path score = 387.41558837890625
  2020-02-27 11:09:22,737 : INFO :   Eval No.2 path score = 54.301971435546875
  2020-02-27 11:09:22,737 : INFO :   Eval No.3 path score = 658.7189331054688
  2020-02-27 11:09:22,737 : INFO :   Eval No.4 path score = 648.3048706054688
  ```

* `crf&init_transitions&multi_decode`

  ```shell
  2020-02-27 11:17:15,118 : INFO : ***** Eval results  *****
  2020-02-27 11:17:15,118 : INFO :   f1 = 0.5776566757493188
  2020-02-27 11:17:15,118 : INFO :   loss = 3.036368516775278
  2020-02-27 11:17:15,118 : INFO :   precision = 0.462882096069869
  2020-02-27 11:17:15,118 : INFO :   recall = 0.7681159420289855
  2020-02-27 11:17:15,118 : INFO : ***** Eval path score(s) in CRF *****
  2020-02-27 11:17:15,118 : INFO :   Eval No.0 path score = [547.3997802734375, 547.3592529296875]
  2020-02-27 11:17:15,118 : INFO :   Eval No.1 path score = [387.41558837890625, 386.2406005859375]
  2020-02-27 11:17:15,118 : INFO :   Eval No.2 path score = [54.301971435546875, 53.47770309448242]
  2020-02-27 11:17:15,118 : INFO :   Eval No.3 path score = [658.7189331054688, 658.0803833007812]
  2020-02-27 11:17:15,118 : INFO :   Eval No.4 path score = [648.3048706054688, 647.9928588867188]
  ```

* 结论：

  - 开启`crf`后，`f1`得到了提升
  - 开启`crf_multi_decode`，`recall`得到了提升

