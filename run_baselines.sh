env_name='halfcheetah-medium-replay-v2'
agent='STR'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='TD3BC'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='TEST'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='AWR'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 
        
        
        
        
    
env_name='hopper-medium-replay-v2'
agent='STR'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='TD3BC'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='TEST'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='AWR'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 





env_name='walker2d-medium-replay-v2'
agent='STR'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='TD3BC'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='TEST'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='AWR'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 
