#!/usr/bin/env bash

max=10
seen_index=0
lr=0.0001
lps=10001
det=True

for i in $(seq 1 $max); do
  python main_mnist.py \
    --seed=$i \
    --algorithm=MEADA \
    --lr=$lr \
    --num_classes=10 \
    --test_every=100 \
    --logs='mnist/logs_'$i'/' \
    --batch_size=32 \
    --model_path='mnist/models_'$i'/' \
    --seen_index=$seen_index \
    --loops_train=$lps \
    --loops_min=100 \
    --step_size=$lps \
    --deterministic=$det \
    --k=3 \
    --eta=10.0
done
