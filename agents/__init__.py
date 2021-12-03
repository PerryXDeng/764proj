from agents.agent_partae import AgentPartAE
from agents.agent_seq2seq import Seq2SeqAgent


# get relevant agent for use
def get_agent(name, config):
    if name == "partae":
        return AgentPartAE(config)
    elif config.module == 'seq2seq':
        return Seq2SeqAgent(config)
    else:
        return False
