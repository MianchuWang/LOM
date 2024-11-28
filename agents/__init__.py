

def return_agent(**agent_params):
    if agent_params['agent'].startswith('BC'):
        from agents.bc import BC
        return BC(**agent_params)
    elif agent_params['agent'].startswith('LOM'):
        from agents.lom import LOM
        return LOM(**agent_params)
    else:
        raise Exception('Invalid agent!')