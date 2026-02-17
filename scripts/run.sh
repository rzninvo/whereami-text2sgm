#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

CUDA_VISIBLE_DEVICES=0 python3 -m whereami.models.train \
    train.epoch=100 \
    train.lr=0.0001 \
    train.weight_decay=0.00005 \
    train.batch_size=16 \
    train.contrastive_loss=true \
    train.training_with_cross_val=true \
    train.folds=10 \
    train.skip_k_fold=true \
    train.entire_training_set=true \
    train.subgraph_ablation=true \
    eval.eval_iters=100 \
    'eval.valid_top_k=[1,2,3,5]' \
    train.model_name=10_06_25_model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint
