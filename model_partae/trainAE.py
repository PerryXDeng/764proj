from agents import getAgent
from tqdm import tqdm
from collections import OrderedDict


def trainAE(config, trainData, valData):

    # load the correct agent
    agent = getAgent("partae", config)

    # load checkpoint if needed
    if config.cont:
        agent.loadChkPt(config.ckpt)
    # set up the clock
    clock = agent.clock

    # iterate over the given epochs
    for e in range(clock.epoch, config.epochs):
        # begin iteration
        pbar = tqdm(trainData)

        for i, data in enumerate(pbar):
            outputs, losses = agent.trainFunc(data)

            pbar.set_description("EPOCH[{}][{}]".format(e, i))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            clock.tick()

        agent.updateLearningRate()
        clock.tock()

    return True
