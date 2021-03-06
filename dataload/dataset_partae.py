from torch.utils.data import Dataset
from dataload.data_utils import getIdsMode, loadH5Partwise
import json
import os
import torch
import randomtest
import numpy as np


# loading the dataset for chair, based on mode selected between train, validate and test
class PartAEDataset(Dataset):
    def __init__(self, mode, config, useAllPts=False):
        super(PartAEDataset, self).__init__()
        self.srcDir = config.srcDir
        self.mode = mode
        self.ptsBatchSize = config.ptsBatchSize
        self.allPts = useAllPts
        self.partsInfo = self.loadPartInfoData(self.mode)
        self.resolution = config.resolution

    def loadPartInfoData(self, mode):
        # get ids of all chair models for this mode
        listIDs = getIdsMode(mode)

        # load the parts dictionary
        with open('data/Chair_info.json', 'r') as dictFile:
            npartsDict = json.load(dictFile)

        partsInfo = []
        for ID in listIDs:
            shapePath = os.path.join(self.srcDir, ID + '.h5')
            if not os.path.exists(shapePath):
                continue
            partsInfo.extend([(shapePath, x) for x in range(npartsDict[ID])])

        return partsInfo

    def __getitem__(self, index):
        shapePath, partInd = self.partsInfo[index]
        nParts, partVoxel, dataPts, dataVals, scales, translations, size = loadH5Partwise(shapePath, partInd,
                                                                                          self.resolution)
        affine = np.concatenate([translations, size])

        # shuffle selected points
        if not self.allPts and len(dataPts) > self.ptsBatchSize:
            indices = np.arange(len(dataPts))
            randomtest.shuffle(indices)
            # np.random.shuffle(indices)
            indices = indices[:self.ptsBatchSize]
            dataPts = dataPts[indices]
            dataVals = dataVals[indices]

        batchVoxels = torch.tensor(partVoxel.astype(np.float), dtype=torch.float32).unsqueeze(0)  # (1, dim, dim, dim)
        batchPoints = torch.tensor(dataPts, dtype=torch.float32)  # (points_batch_size, 3)
        batchValues = torch.tensor(dataVals, dtype=torch.float32)  # (points_batch_size, 1)

        return {"vox3d": batchVoxels,
                "points": batchPoints,
                "values": batchValues,
                "n_parts": nParts,
                "part_idx": partInd,
                "path": shapePath,
                "scales": scales,
                "translations": translations,
                "size": size,
                "affine": affine}

    def __len__(self):
        return len(self.partsInfo)


if __name__ == "__main__":
    pass
