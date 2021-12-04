from agents.agent_partae import AgentPartAE


# get relevant agent for use
def get_agent(name, config):
    return AgentPartAE(config)
