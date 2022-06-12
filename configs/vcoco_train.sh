#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --project_name CPC_QPIC_VCOCO \
    --run_name ${EXP_DIR} \
    --pretrained params/detr-r50-pre-vcoco-cpc.pth \
    --hoi \
    --epochs 80 \
    --lr_drop 60 \
    --lr 1e-4 \
    --batch_size 2 \
    --lr_backbone 1e-5 \
    --ramp_up_epoch 30 \
    --ramp_down_epoch 50 \
    --hoi_consistency_loss_coef 0.1 \
    --path_id 0 \
    --dataset_file vcoco \
    --hoi_path data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --backbone resnet50 \
    --reg_consistency_loss_coef 2.5 \
    --obj_consistency_loss_coef 1 \
    --verb_consistency_loss_coef 2 \
    --output_dir checkpoints/vcoco/ \
    --augpath_name [\'p2\',\'p3\',\'p4\'] \
    ${PY_ARGS}
    