from agents.agent_partae import AgentPartAE


def getAgent(name, config):
    if name == "partae":
        agentRet = AgentPartAE(config)
    else:
        return False
    return agentRet
