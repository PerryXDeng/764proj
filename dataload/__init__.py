import numpy as np
from torch.utils.data import DataLoader
from dataload.dataset import MainDataset


def getDataLoader(config, phase, shuffle):

    datasetMain = MainDataset(phase, config)
    dataloader = DataLoader(datasetMain, batch_size=config.batchSize, shuffle=shuffle,
                            num_workers=config.num_workers, worker_init_fn=np.random.seed())

    return dataloader
