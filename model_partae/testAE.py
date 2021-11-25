from agents import getAgent
from tqdm import tqdm
import torch
import numpy as np
from utils import cycle
# since partae works with parts only the output will be parts, and not the whole object


# encode using partae
def reconstruct(config, testData):
    agent = getAgent("partae")

    testData = cycle(testData)
    data = next(testData)

    agent.visualizeCurBatch(data, "test")


def testAE(config, testData):
    # encode and decode the test files only
    agent = getAgent("partae", config)

    # save them as mesh
    config.saveDir = "result/partae"
    config.saveFormat = "mesh"

    reconstruct(config, testData)

    return False
