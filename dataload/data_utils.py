import os
import json
import h5py
import numpy as np

SplitDir = "data/train_val_test_split"


# loads IDs for models for modes = train,val or test
def getIdsMode(mode):
    filename = os.path.join(SplitDir, "{}.{}.json".format("Chair", mode))
    if not os.path.exists(filename):
        raise ValueError("Invalid Path in Dataload " + filename)

    with open(filename) as json_file:
        data = json.load(json_file)

    listIds = []
    for item in data:
        listIds.append(item["anno_id"])

    return listIds


# load part wise information
def loadH5Partwise(path, partInd, resolution=64, rescale=True):
    with h5py.File(path, 'r') as data_dict:
        nParts = data_dict.attrs['n_parts']
        partVoxel = data_dict['parts_voxel_scaled64'][partInd].astype(np.float)
        dataPoints = data_dict['points_{}'.format(resolution)][partInd]
        dataVals = data_dict['values_{}'.format(resolution)][partInd]
    if rescale:
        dataPoints = dataPoints / resolution
    return nParts, partVoxel, dataPoints, dataVals  # , scale, translation
