#!/bin/bash

envs=('halfcheetah-medium-v2' 'hopper-medium-v2' 'walker2d-medium-v2' 'halfcheetah-medium-replay-v2' 'hopper-medium-replay-v2' 'walker2d-medium-replay-v2' 'halfcheetah-medium-expert-v2' 'hopper-medium-expert-v2' 'walker2d-medium-expert-v2' 'halfcheetah-expert-v2' 'hopper-expert-v2' 'walker2d-expert-v2' 'halfcheetah-random-v2' 'hopper-random-v2' 'walker2d-random-v2')
num_gpus=$(nvidia-smi -L | wc -l)
num_envs=${#envs[@]}

for ((i=0; i<num_envs; i++)); do
    CUDA_VISIBLE_DEVICES=$((i % num_gpus)) python train.py --env_name ${envs[i]} &
done



