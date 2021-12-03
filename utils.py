import numpy as np
from networks import getNetwork
import torch.nn as nn
import mcubes as libmcubes
import trimesh
import os


class TrainClock(object):
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def makeChkPt(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restoreChkPt(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


# get the reqwuired network from
def buildNet(name, config):
    net = getNetwork(name, config)
    if config.parallel:
        net = nn.DataParallel(net)
    net = net.cuda()
    return net


def projVoxelXYZ(voxels, concat=False):
    # projecting the outline or max in each direction on its specific axis
    img1 = np.clip(np.amax(voxels, axis=0) * 256, 0, 255).astype(np.uint8)
    img2 = np.clip(np.amax(voxels, axis=1) * 256, 0, 255).astype(np.uint8)
    img3 = np.clip(np.amax(voxels, axis=2) * 256, 0, 255).astype(np.uint8)
    if concat:
        dim = img1.shape[0]
        line = np.zeros((dim, 2), dtype=np.uint8)
        wholeImg = np.concatenate([img1, line, img2, line, img3], axis=1)
        return wholeImg
    else:
        return img1, img2, img3


def voxel2mesh(voxels, config, export=False, name='mymodel', affine=None, voxDim=64,
               idx=None, size=None, translation=None, inCol=None, scale=None):
    vertices, triangles = libmcubes.marching_cubes(voxels, 0)

    # if affine is not None:
    #     vertices = vertices * affine[0, :] + affine[1, :] * voxDim

    mesh = trimesh.Trimesh(vertices, triangles, face_colors=inCol)

    mesh.apply_translation((-32, -32, -32))
    mesh.apply_scale([scale, scale, scale])
    mesh.apply_translation(translation[:])

    if export:
        savePath = os.path.join(config.saveDir, "model_{}.stl".format(name))
        mesh.export(savePath)

    return mesh


def sdf2voxel(points, values, voxDim=64):
    points = np.round(points).astype(int)
    voxels = np.zeros([voxDim, voxDim, voxDim], np.uint8)
    discrete_values = np.zeros_like(values, dtype=np.uint8)
    discrete_values[np.where(values > 0.5)] = 1
    voxels[points[:, 0], points[:, 1], points[:, 2]] = np.reshape(discrete_values, [-1])
    return voxels


def visualizeSDF(points, values, concat=False, voxDim=64):
    voxels = sdf2voxel(points, values, voxDim)
    return projVoxelXYZ(voxels, concat=concat)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
