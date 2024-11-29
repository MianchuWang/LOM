#!/bin/bash

envs=(
'halfcheetah-medium-replay-v2' 
'hopper-medium-replay-v2' 
'walker2d-medium-replay-v2'

'halfcheetah-full-replay-v2' 
'hopper-full-replay-v2' 
'walker2d-full-replay-v2'

'halfcheetah-medium-v2'
'hopper-medium-v2'
'walker2d-medium-v2'

'halfcheetah-expert-v2'
'hopper-expert-v2'
'walker2d-expert-v2'

'halfcheetah-medium-expert-v2'
'hopper-medium-expert-v2'
'walker2d-medium-expert-v2'
)

num_gpus=$(nvidia-smi -L | wc -l)
i=0

for env in "${envs[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % num_gpus)) python train.py --env_name "$env"
    i=$((i + 1))
done

wait
