#!/usr/bin/env bash

max=1
lr=0.1
epochs=100
det=True

for i in $(seq 1 $max); do
  python main_cifar.py \
    --seed=$i \
    --algorithm=MEADA \
    --model=wrn \
    --dataset=cifar10 \
    --epochs=$epochs \
    --batch_size=128 \
    --lr=$lr \
    --logs='cifar10/logs_'$i'/' \
    --model_path='cifar10/models_'$i'/' \
    --deterministic=$det \
    --gamma=1.0 \
    --k=2 \
    --epochs_min=10 \
    --eta=10.0
done
