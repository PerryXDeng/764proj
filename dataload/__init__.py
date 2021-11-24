import numpy as np
from torch.utils.data import DataLoader
from dataload.dataset import MainDataset


def getDataLoader(mode, config, useAllPts, shuffle):

    datasetMain = MainDataset(mode, config, useAllPts)
    dataloader = DataLoader(datasetMain, batch_size=config.batchSize, shuffle=shuffle,num_workers=config.numWorkers, worker_init_fn=np.random.seed())

    return dataloader
