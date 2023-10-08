from agents.bc import BC
from agents.str import STR
from agents.td3bc import TD3BC


def return_agent(**agent_params):
    if agent_params['agent'] == 'bc':
        return BC(**agent_params)
    elif agent_params['agent'] == 'str':
        return STR(**agent_params)
    elif agent_params['agent'] == 'td3bc':
        return TD3BC(**agent_params)
    else:
        raise Exception('Invalid agent!')