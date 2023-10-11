env_name='halfcheetah-medium-replay-v2'
agent='str'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent}
        
env_name='hopper-medium-replay-v2'
agent='str'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent}


env_name='walker2d-medium-replay-v2'
agent='str'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent}







env_name='halfcheetah-medium-v2'
agent='str'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent}
        
env_name='hopper-medium-v2'
agent='str'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent}


env_name='walker2d-medium-v2'
agent='str'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='td3bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='bc'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent} &

agent='sota'
python train.py --env_name ${env_name} --agent ${agent} &
python train.py --env_name ${env_name} --agent ${agent}
