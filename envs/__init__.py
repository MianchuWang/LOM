import gym
import d4rl


mujoco_locomotion = ['halfcheetah-medium-v2', 'hopper-medium-v2', 'walker2d-medium-v2']




def return_environment(env_name):
    if env_name in mujoco_locomotion:
        env = gym.make(env_name)
        env_info = {'state_dim': env.observation_space.shape[0],
                    'ac_dim': env.action_space.shape[0]}
        return env, env_info
