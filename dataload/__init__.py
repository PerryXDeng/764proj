from torch.utils.data import DataLoader
import numpy as np
from dataload.dataset_partae import PartAEDataset
from dataload.dataset_seq2seq import Seq2SeqDataset, padCollateFNDict


def get_dataloader(mode, config, useAllPts=True, shuffle=False):
    shuffle = mode == 'train' if shuffle is None else shuffle

    if config.module == 'part_ae':
        datasetMain = PartAEDataset(mode, config, useAllPts)
        dataLoader = DataLoader(datasetMain, batch_size=config.batchSize, shuffle=shuffle, num_workers=config.numWorkers
                                , worker_init_fn=np.random.seed())
    elif config.module == 'seq2seq':
        dataset = Seq2SeqDataset(mode, config.dataRoot, config.maxNumParts)
        dataLoader = DataLoader(dataset, batch_size=config.batchSize, shuffle=shuffle, num_workers=config.numWorkers,
                                collate_fn=padCollateFNDict)
    else:
        return False

    return dataLoader
