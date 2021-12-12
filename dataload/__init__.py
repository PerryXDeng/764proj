from torch.utils.data import DataLoader
import numpy as np
from dataload.dataset_partae import PartAEDataset

def get_dataloader(mode, config, useAllPts=True, shuffle=False):
    shuffle = mode == 'train' if shuffle is None else shuffle

    datasetMain = PartAEDataset(mode, config, useAllPts)
    dataLoader = DataLoader(datasetMain, batch_size=config.batchSize, shuffle=shuffle, num_workers=config.numWorkers
                            , worker_init_fn=np.random.seed())

    return dataLoader
