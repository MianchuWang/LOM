mujoco_locomotion = ['halfcheetah-medium-v2', 
                     'hopper-medium-v2', 
                     'walker2d-medium-v2',
                     'halfcheetah-medium-replay-v2', 
                     'hopper-medium-replay-v2', 
                     'walker2d-medium-replay-v2',
                     'halfcheetah-medium-expert-v2',
                     'hopper-medium-expert-v2',
                     'walker2d-medium-expert-v2',
                     'halfcheetah-expert-v2',
                     'hopper-expert-v2',
                     'walker2d-expert-v2',
                     'halfcheetah-random-v2',
                     'hopper-random-v2',
                     'walker2d-random-v2',
                     'halfcheetah-full-replay-v2',
                     'hopper-full-replay-v2',
                     'walker2d-full-replay-v2']



def return_environment(env_name):
    if env_name in mujoco_locomotion:
        try:
            import gym
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'Gym is required for MuJoCo locomotion environments. '
                'Install the project dependencies in a Python 3.9 environment.'
            ) from exc
        try:
            import d4rl
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                'D4RL is required for MuJoCo locomotion environments. '
                'Install d4rl, gym, and mujoco-py in the Python 3.9 environment '
                'specified by the README.'
            ) from exc
        env = gym.make(env_name)
        env_info = {'state_dim': env.observation_space.shape[0],
                    'ac_dim': env.action_space.shape[0]}
        return env, env_info
    raise ValueError('Unknown environment: {}'.format(env_name))
