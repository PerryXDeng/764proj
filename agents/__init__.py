from agents.agent_partae import AgentPartAE


# get relevant agent for use
def getAgent(name, config):
    if name == "partae":
        agentRet = AgentPartAE(config)
        return agentRet
    else:
        return False

