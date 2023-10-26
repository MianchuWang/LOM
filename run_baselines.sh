env_name='halfcheetah-medium-replay-v2'
agent='str'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='test'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='awr'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 
        
        
        
        
    
env_name='hopper-medium-replay-v2'
agent='str'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='test'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='awr'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 





env_name='walker2d-medium-replay-v2'
agent='str'
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='test'
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='awr'
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} 
