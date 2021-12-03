from agents import get_agent
from tqdm import tqdm
from collections import OrderedDict
from utils import cycle


def train(config, trainData, valData):
    agent = get_agent("seq2seq", config)
    config.cont = "False"
    # load from checkpoint if provided
    if config.cont:
        agent.load_ckpt(config.ckpt)

    # start training
    clock = agent.clock

    for e in range(clock.epoch, config.nr_epochs):
        # begin iteration
        pbar = tqdm(trainData)
        for b, data in enumerate(pbar):
            # train step
            outputs, losses = agent.train_func(data)

            # visualize
            if config.vis and clock.step % config.vis_frequency == 0:
                agent.visualize_batch(data, 'train', outputs=outputs)

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % config.val_frequency == 0:
                data = next(valData)
                outputs, losses = agent.val_func(data)

                if config.vis and clock.step % config.vis_frequency == 0:
                    agent.visualize_batch(data, 'validation', outputs=outputs)

            clock.tick()

        # update lr by scheduler
        agent.update_learning_rate()

        # update teacher forcing ratio
        if config.module == 'seq2seq':
            agent.update_teacher_forcing_ratio()

        clock.tock()
        if clock.epoch % config.save_frequency == 0:
            agent.save_ckpt()
        agent.save_ckpt('latest')

    return True
