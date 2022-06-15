#!/usr/bin/env bash

set -x

EXP_DIR=logs_run_001
PY_ARGS=${@:1}

python -u main.py \
    --pretrained params/detr-r50-pre-hico-cpc.pth \
    --run_name ${EXP_DIR} \
    --project_name CPC_QPIC_HICODET \
    --hoi \
    --epochs 80 \
    --lr_drop 50 \
    --use_consis \
    --share_dec_param \
    --stop_grad_stage \
    --ramp_up_epoch 30 \
    --hoi_consistency_loss_coef 0.2 \
    --path_id 0 \
    --augpath_name [\'p2\',\'p3\',\'p4\'] \
    --dataset_file hico \
    --hoi_path data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --reg_consistency_loss_coef 2.5 \
    --verb_consistency_loss_coef 1 \
    --obj_consistency_loss_coef 1 \
    --output_dir checkpoints/hicodet/ \
    ${PY_ARGS}
    