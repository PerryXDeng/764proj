from torch.utils.data import DataLoader
import numpy as np
from dataload.dataset import MainDataset


def get_dataloader(mode, config, useAllPts, shuffle):
    datasetMain = MainDataset(mode, config, useAllPts)
    dataLoader = DataLoader(datasetMain, batch_size=config.batchSize, shuffle=shuffle, num_workers=config.numWorkers,
                            worker_init_fn=np.random.seed())

    return dataLoader
