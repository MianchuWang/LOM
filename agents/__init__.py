from agents.bc import BC
from agents.str import STR
from agents.td3bc import TD3BC
from agents.sota import SOTA


def return_agent(**agent_params):
    if agent_params['agent'].startswith('bc'):
        return BC(**agent_params)
    elif agent_params['agent'].startswith('str'):
        return STR(**agent_params)
    elif agent_params['agent'].startswith('td3bc'):
        return TD3BC(**agent_params)
    elif agent_params['agent'].startswith('sota'):
        return SOTA(**agent_params)
    else:
        raise Exception('Invalid agent!')