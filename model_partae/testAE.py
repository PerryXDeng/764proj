from agents import getAgent
from tqdm import tqdm
import numpy as np
from utils import cycle, sdf2voxel, voxel2mesh
import torch
import trimesh
import os


# output all parts of output chair
def outputWholeChair(agent, config, testData):
    config.ckpt = 'latest'
    # load check point
    if config.cont:
        agent.loadChkPt(config.ckpt)

        testData = cycle(testData)
        data = next(testData)
        numShapes = 10




        for j in range(numShapes):
            nParts = data['n_parts']  # (shape_batch_size, 1)
            shape_mesh = []
            for i in range(nParts):

                voxDim = 16

                points = data['points'][0].numpy() * voxDim

                translation = data['translations'][:]
                size = data['size'][:]
                affine = np.concatenate([translation, size])
                # outputs, losses = agent.valFunc(data)
                # values = outputs[0].detach().cpu().numpy()
                targetSDF = data['values'][0].numpy()

                voxels = sdf2voxel(points, targetSDF, voxDim=voxDim)
                mesh = voxel2mesh(voxels, config, export=False, name="", translation=None, size=None,
                                  affine=affine, voxDim=voxDim, idx=i)

                shape_mesh.append(mesh)
                data = next(testData)

            shape_mesh = trimesh.util.concatenate(shape_mesh)
            savePath = os.path.join(config.saveDir, "model_{}.stl".format(j))
            shape_mesh.export(savePath)


# make a code for the Mixer [latent code, scale, translation]

def makeCode(agent, config, testData):
    config.ckpt = 'latest'
    # load check point
    if config.cont:
        agent.loadChkPt(config.ckpt)

    testData = cycle(testData)
    data = next(testData)
    numShapes = 1
    for j in range(numShapes):
        nParts = data['n_parts']  # (shape_batch_size, 1)
        for i in range(nParts):
            inVox3d = data['vox3d'].cuda()
            encOut = agent.net.encoder(inVox3d).cpu().detach().numpy().flatten()
            translation = data['translations'].cpu().detach().numpy().flatten()
            scale = data['scales'].cpu().detach().numpy().flatten()
            totalCode = [encOut, scale, translation]

            print(totalCode)

            data = next(testData)


# get a 3d reconstruction out of the given test data
# since partae works with parts only the output will be parts, and not the whole object
def reconstruct(agent, config, testData):
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

        voxelsFinal = sdf2voxel(data_points64, output_sdf)
        mesh = voxel2mesh(voxelsFinal, config, export=True, name="_edit" + str(i))


def testAE(config, testData):
    # encode and decode the test files only
    agent = getAgent("partae", config)

    # save them as mesh
    config.saveDir = "results/partae"
    config.saveFormat = "mesh"

    # reconstruct(agent, config, testData)
    # makeCode(agent, config, testData)
    outputWholeChair(agent, config, testData)

    return False
