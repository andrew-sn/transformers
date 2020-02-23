## BERT_CRF

### 最佳实践

* 开启`CRF`会降低GPU的使用效率，因此在大型任务上建议不开启；
* 通过设置`crf_decode_topk > 1`可以开启`crf_multi_decode`，在更重视`Recall`的任务场景中可以使用；

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
  2020-02-23 16:04:49,555 : INFO : ***** Eval results  *****
  2020-02-23 16:04:49,556 : INFO :   f1 = 0.5088757396449703
  2020-02-23 16:04:49,556 : INFO :   loss = 7.436665755051833
  2020-02-23 16:04:49,556 : INFO :   precision = 0.43
  2020-02-23 16:04:49,556 : INFO :   recall = 0.6231884057971014
  2020-02-23 16:04:49,556 : INFO : ***** Eval path score(s) in CRF *****
  2020-02-23 16:04:49,556 : INFO :   Eval No.0 path score = 721.51806640625
  2020-02-23 16:04:49,556 : INFO :   Eval No.1 path score = 524.163818359375
  2020-02-23 16:04:49,556 : INFO :   Eval No.2 path score = 66.89285278320312
  2020-02-23 16:04:49,556 : INFO :   Eval No.3 path score = 867.07080078125
  2020-02-23 16:04:49,557 : INFO :   Eval No.4 path score = 875.7245483398438
  ```

* `crf&init_transitions&multi_decode`

  ```shell
  2020-02-23 16:06:54,169 : INFO : ***** Eval results  *****
  2020-02-23 16:06:54,170 : INFO :   f1 = 0.4198645598194131
  2020-02-23 16:06:54,170 : INFO :   loss = 7.436665755051833
  2020-02-23 16:06:54,170 : INFO :   precision = 0.30491803278688523
  2020-02-23 16:06:54,170 : INFO :   recall = 0.6739130434782609
  2020-02-23 16:06:54,170 : INFO : ***** Eval path score(s) in CRF *****
  2020-02-23 16:06:54,170 : INFO :   Eval No.0 path score = [721.51806640625, 720.9774169921875]
  2020-02-23 16:06:54,171 : INFO :   Eval No.1 path score = [524.163818359375, 523.3126220703125]
  2020-02-23 16:06:54,171 : INFO :   Eval No.2 path score = [66.89285278320312, 66.7175521850586]
  2020-02-23 16:06:54,171 : INFO :   Eval No.3 path score = [867.07080078125, 866.3570556640625]
  2020-02-23 16:06:54,171 : INFO :   Eval No.4 path score = [875.7245483398438, 875.4892578125]
  ```

* 结论：

  - 开启`crf`后，`f1`得到了提升
  - 开启`crf_multi_decode`，`recall`得到了提升