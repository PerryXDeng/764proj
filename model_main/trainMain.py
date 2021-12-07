from agents import get_agent
from tqdm import tqdm
from collections import OrderedDict
from utils import cycle


def train(config, trainData, valData):
    agent = get_agent("seq2seq", config)

    # load from checkpoint if provided
    if config.cont:
        agent.loadChkPt(config.ckpt)

    # start training
    clock = agent.clock

    for e in range(clock.epoch, config.epochs):
        # begin iteration
        pbar = tqdm(trainData)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = agent.trainFunc(data)

            # visualize
            if config.vis and clock.step % config.visFrequency == 0:
                agent.visualize_batch(data, 'train', outputs=outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % config.valFrequency == 0:
                data = next(valData)
                outputs, losses = agent.valFunc(data)

                if config.vis and clock.step % config.visFrequency == 0:
                    agent.visualize_batch(data, 'validation', outputs=outputs)

            clock.tick()

        # update lr by scheduler
        agent.updateLearningRate()

        # update teacher forcing ratio
        if config.module == 'seq2seq':
            agent.update_teacher_forcing_ratio()

        clock.tock()
        if clock.epoch % config.saveFrequency == 0:
            agent.saveChkPt()
        agent.saveChkPt('latest')

    return True
