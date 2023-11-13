env_name='halfcheetah-medium-v2'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='hopper-medium-v2'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='walker2d-medium-v2'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='halfcheetah-medium-replay-v2'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='hopper-medium-replay-v2'
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='walker2d-medium-replay-v2'
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='halfcheetah-medium-expert-v2'
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='hopper-medium-expert-v2'
cuda=7
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='walker2d-medium-expert-v2'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='halfcheetah-expert-v2'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='hopper-expert-v2'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='walker2d-expert-v2'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='halfcheetah-random-v2'
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='hopper-random-v2'
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &

env_name='walker2d-random-v2'
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} &
