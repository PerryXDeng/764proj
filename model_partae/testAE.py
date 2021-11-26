from agents import getAgent
from tqdm import tqdm
import numpy as np
from utils import cycle, sdf2voxel, voxel2mesh


# since partae works with parts only the output will be parts, and not the whole object


# encode using partae
def reconstruct(config, testData):
    agent = getAgent("partae", config)

    config.ckpt = 'latest'
    # load check point
    if config.cont:
        agent.loadChkPt(config.ckpt)

    testData = cycle(testData)

    for i in range(10):
        data = next(testData)
        data_points64 = data['points'][0].numpy() * config.resolution
        targetSDF = data['values'][0].numpy()
        outputs, losses = agent.valFunc(data)
        output_sdf = outputs[0].detach().cpu().numpy()

        voxelsFinal = sdf2voxel(data_points64, targetSDF)
        mesh = voxel2mesh(voxelsFinal, config, export=True, name=str(i))


def testAE(config, testData):
    # encode and decode the test files only
    agent = getAgent("partae", config)

    # save them as mesh
    config.saveDir = "results/partae"
    config.saveFormat = "mesh"

    reconstruct(config, testData)

    return False
