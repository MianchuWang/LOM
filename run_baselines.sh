cuda=0

env_name='halfcheetah-medium-replay-v2'
agent='str'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent}
        
env_name='hopper-medium-replay-v2'
agent='str'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent}


env_name='walker2d-medium-replay-v2'
agent='str'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent}






env_name='halfcheetah-medium-v2'
agent='str'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent}
        
env_name='hopper-medium-v2'
agent='str'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent}


env_name='walker2d-medium-v2'
agent='str'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent}
