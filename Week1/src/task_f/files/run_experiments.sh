#!/bin/bash

DATASET=/DATA/home/jgarcia/SpectralSegmentation/C5-Project/Week1/DEArt_dataset/DEArt_dataset

echo "Run 1 - Baseline"
python -m src.task_f.main_train \
--dataset $DATASET \
--variant resnet50_fpn_v2 \
--batch_size 6 \
--lr 0.0005 \
--epochs 20 \
--aug None

echo "Run 2 - Mild Augmentation"
python -m src.task_f.main_train \
--dataset $DATASET \
--variant resnet50_fpn_v2 \         
--batch_size 6 \
--lr 0.0005 \
--epochs 20 \
--aug mild

echo "Run 3 - Aggressive Augmentation"
python -m src.task_f.main_train \
--dataset $DATASET \
--variant resnet50_fpn_v2 \
--batch_size 6 \
--lr 0.0005 \
--epochs 20 \
--aug aggressive

echo "Run 4 - Freeze Backbone"
python -m src.task_f.main_train \
--dataset $DATASET \
--variant resnet50_fpn_v2 \
--batch_size 6 \
--lr 0.0005 \
--epochs 20 \
--aug mild \
--freeze_backbone 

echo "Run 6 -  Balanced Sampling +  More Epochs"
python -m src.task_f.main_train \
--dataset $DATASET \
--variant resnet50_fpn_v2 \
--batch_size 8 \
--lr 0.0005 \
--epochs 30 \
--aug aggressive \

