

def return_agent(**agent_params):
    if agent_params['agent'].startswith('BC'):
        from agents.bc import BC
        return BC(**agent_params)
    elif agent_params['agent'].startswith('STR'):
        from agents.str import STR
        return STR(**agent_params)
    elif agent_params['agent'].startswith('TD3BC'):
        from agents.td3bc import TD3BC
        return TD3BC(**agent_params)
    elif agent_params['agent'].startswith('AWR'):
        from agents.awr import AWR
        return AWR(**agent_params)
    elif agent_params['agent'].startswith('WCGAN'):
        from agents.wcgan import WCGAN
        return WCGAN(**agent_params)
    elif agent_params['agent'].startswith('GMM'):
        from agents.gmm import GMM
        return GMM(**agent_params)
    elif agent_params['agent'].startswith('seqGMM'):
        from agents.seq_gmm import seqGMM
        return seqGMM(**agent_params)
    else:
        raise Exception('Invalid agent!')