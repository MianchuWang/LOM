env_name='halfcheetah-medium-replay-v2'
agent='EXPLO-lamb0'
lamb=0
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb25'
lamb=0.25
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb50'
lamb=0.5
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=7
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb75'
lamb=0.75
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb10'
lamb=1
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} 







env_name='hopper-medium-replay-v2'
agent='EXPLO-lamb0'
lamb=0
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb25'
lamb=0.25
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb50'
lamb=0.5
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=7
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb75'
lamb=0.75
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb10'
lamb=1
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb}






env_name='walker2d-medium-replay-v2'
agent='EXPLO-lamb0'
lamb=0
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb25'
lamb=0.25
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb50'
lamb=0.5
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=7
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=0
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb75'
lamb=0.75
cuda=1
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=3
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &

agent='EXPLO-lamb10'
lamb=1
cuda=4
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb} &
cuda=6
CUDA_VISIBLE_DEVICES=${cuda} python train.py --env_name ${env_name} --agent ${agent} --lamb ${lamb}

