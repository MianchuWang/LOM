import numpy as np
import gym
import d4rl
gym.logger.set_level(40)

gym_robotics = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1',
                'FetchSlide-v1', 'HandReach-v0']
d4rl_antmaze_envs = ['antmaze-umaze-v2', 'antmaze-umaze-diverse-v2',
                     'antmaze-medium-play-v2', 'antmaze-medium-diverse-v2',
                     'antmaze-large-play-v2', 'antmaze-large-diverse-v2']


def get_goal_from_state(env_name):
    if env_name.startswith('FetchReach'):
        return lambda x : x[..., :3]
    elif env_name.startswith('FetchPush'):
        return lambda x : x[..., 3:6]
    elif env_name.startswith('FetchPick'):
        return lambda x : x[..., 3:6]
    elif env_name.startswith('FetchSlide'):
        return lambda x : x[..., 3:6]
    elif env_name.startswith('HandReach'):
        return lambda x : x[..., -15:]
    else:
        raise Exception('Invalid environment. The environments options are', 
                        gym_robotics + d4rl_antmaze_envs)

def return_environment(env_name, render_mode):
    if env_name in gym_robotics:
        return return_gym_robotics_env(env_name, render_mode)
    elif env_name in d4rl_antmaze_envs:
        return return_d4rl_env(env_name, render_mode)
    else:
        raise Exception('Invalid environment.')
    
def return_gym_robotics_env(env_name, render_mode):
    # import gymnasium as gym
    
    class GymWrapper(gym.RewardWrapper):
        def reward(self, reward):
            # return = 1 if success else 0
            return reward + 1
    
    env = gym.make(env_name)
    return GymWrapper(env), \
           {'env_name': env_name,
            'state_dim': env.observation_space['observation'].shape[0],
            'goal_dim': env.observation_space['desired_goal'].shape[0],
            'ac_dim': env.action_space.shape[0],
            'max_steps': 50,
            'get_goal_from_state': get_goal_from_state(env_name),
            'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}
    

def return_d4rl_env(env_name, render_mode):
    class AntWrapper(gym.ObservationWrapper):
        ''' Wrapper for exposing the goals of the AntMaze environment. '''
        def observation(self, observation):
            return {'observation': observation, 
                    'achieved_goal': observation[:2],
                    'desired_goal': np.array(self.env.target_goal)}
        def compute_reward(self, achieved_goal, desired_goal, info):
            reward = (np.linalg.norm(achieved_goal - desired_goal, axis=-1) <= 0.5).astype(np.float32)
            return reward
        @property
        def max_episode_steps(self):
            return self.env._max_episode_steps
    env = gym.make(env_name)
    return AntWrapper(env), {'env_name': env_name,
                             'state_dim': env.observation_space.shape[0],
                             'goal_dim': 2,
                             'ac_dim': env.action_space.shape[0],
                             'max_steps': env._max_episode_steps, 
                             'get_goal_from_state': lambda x : x[..., :2],
                             'compute_reward': lambda x, y, z: (np.linalg.norm(x - y, axis=-1) <= 0.5).astype(np.float32)}



