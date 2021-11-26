from agents import getAgent
from tqdm import tqdm
from collections import OrderedDict
from utils import cycle


def trainAE(config, trainData, valData):
    # load the correct agent
    agent = getAgent("partae", config)

    # load checkpoint if needed
    if config.cont:
        agent.loadChkPt(config.ckpt)
    # set up the clock
    clock = agent.clock

    valData = cycle(valData)

    # iterate over the given epochs
    for e in range(clock.epoch, config.epochs):
        # begin iteration
        pbar = tqdm(trainData)

        for i, data in enumerate(pbar):
            outputs, losses = agent.trainFunc(data)

            # visualize
            if config.vis and clock.step % config.visFrequency == 0:
                agent.visualizeCurBatch(data, 'train', outputs=outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, i))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % config.valFrequency == 0:
                data = next(valData)
                outputs, losses = agent.valFunc(data)

                if config.vis and clock.step % config.visFrequency == 0:
                    agent.visualizeCurBatch(data, 'validation', outputs=outputs)

            clock.tick()

        agent.updateLearningRate()
        clock.tock()

        if clock.epoch % config.saveFrequency == 0:
            agent.saveChkPt()
        agent.saveChkPt('latest')

    return True
