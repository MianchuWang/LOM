

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
    elif agent_params['agent'].startswith('BPI'):
        from agents.bpi import BPI
        return BPI(**agent_params)
    elif agent_params['agent'].startswith('CPI'):
        from agents.cpi import CPI
        return CPI(**agent_params)
    elif agent_params['agent'].startswith('EXPLO'):
        from agents.exploration import EXPLORATION
        return EXPLORATION(**agent_params)
    elif agent_params['agent'].startswith('CVAE'):
        from agents.cvae import CVAE
        return CVAE(**agent_params)
    elif agent_params['agent'].startswith('seqCVAE'):
        from agents.seq_cvae import seqCVAE
        return seqCVAE(**agent_params)
    elif agent_params['agent'].startswith('seqGMM'):
        from agents.seq_gmm import seqGMM
        return seqGMM(**agent_params)
    else:
        raise Exception('Invalid agent!')