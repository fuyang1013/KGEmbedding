# krl

知识图谱嵌入

不确定知识图谱 UKGE

### train 

```
python ukge.py --do_train --train_epochs 100 --num_ents 15000 --num_rels 36 --num_dim 128 --mapping logistic --batch_size 32 --device cuda --lr 0.001 --train_file dataset/cn15k/train.csv --eval_file dataset/cn15k/dev.csv --debug
```
