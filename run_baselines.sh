agent='EXPLO-baw'

env_name='halfcheetah-medium-replay-v2'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &


env_name='hopper-medium-replay-v2'
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=7
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &


env_name='walker2d-medium-replay-v2'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 

