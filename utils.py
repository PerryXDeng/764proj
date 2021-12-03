import numpy as np
from networks import get_network
import torch.nn as nn
import mcubes as libmcubes
import trimesh
import os
from itertools import product, combinations

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
    net = get_network(name, config)
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


def affine2bboxes(affine, limit=64):
    mins = (affine[:, :3] - affine[:, 3:6] / 2) * limit
    maxs = (affine[:, :3] + affine[:, 3:6] / 2) * limit
    bboxes = np.clip(np.round(np.concatenate([mins, maxs], axis=1)).astype(int), 0, limit - 1)
    return bboxes


def drawPartsBBOXVoxel(affine=None, bboxes=None, limit=64, proj=True):
    if bboxes is None:
        bboxes = affine2bboxes(affine, limit)
    n_parts = len(bboxes)
    voxel = np.zeros((limit, limit, limit), dtype=np.uint8)
    for idx in range(n_parts):
        bbox = bboxes[idx]
        size = (bbox[3:] - bbox[:3]).tolist()
        for s, e in combinations(np.array(list(product(bbox[[0, 3]], bbox[[1, 4]], bbox[[2, 5]]))), 2):
            if np.sum(np.abs(s - e)) in size:
                if s[0] != e[0]:
                    voxel[s[0]:e[0], s[1], s[2]] = 1
                elif s[1] != e[1]:
                    voxel[s[0], s[1]:e[1], s[2]] = 1
                else:
                    voxel[s[0], s[1], s[2]:e[2]] = 1
    if proj:
        return projVoxelXYZ(voxel, concat=True)
    return voxel


def partsdf2voxel(points, values, vox_dim=64, by_part=True):
    """

    :param points: (n_parts, n_points, 3) or [(n_points1, 3), (n_points2, 3), ...]
    :param values: (n_parts, n_points, 1) or [(n_points1, 1), (n_points2, 1), ...]
    :return: voxel: (vox_dim, vox_dim, vox_dim)
    """
    n_parts = len(points)
    # points = np.round(points).astype(int)
    voxels = np.zeros([vox_dim, vox_dim, vox_dim], np.uint8)
    for idx in range(n_parts):
        part_points = np.round(points[idx]).astype(int)
        part_values = values[idx]
        postive_points = part_points[np.where(part_values >= 0.5)[0]]
        voxels[postive_points[:, 0], postive_points[:, 1], postive_points[:, 2]] = idx + 1
    if not by_part:
        voxels[np.where(voxels >= 1)] = 1
    return voxels


def partsdf2mesh(points, values, affine=None, vox_dim=64, by_part=True):
    """

    :param points: (n_parts, n_points, 3) or [(n_points1, 3), (n_points2, 3), ...]
    :param values: (n_parts, n_points, 1) or [(n_points1, 1), (n_points2, 1), ...]
    :param affine: (n_parts, 1, 4)
    :param vox_dim: int
    :return:
    """
    # if vox_dim is None:
    #     vox_dim = vox_dim
    if not by_part:
        shape_voxel = partsdf2voxel(points, values, vox_dim=vox_dim, by_part=False)
        vertices, triangles = libmcubes.marching_cubes(shape_voxel, 0)
        shape_mesh = trimesh.Trimesh(vertices, triangles)
        return shape_mesh

    n_parts = len(points)
    colors = [[0, 0, 255, 255],      # blue
              [0, 255, 0, 255],      # green
              [255, 0, 0, 255],      # red
              [255, 255, 0, 255],    # yellow
              [0, 255, 255, 255],    # cyan
              [255, 0, 255, 255],    # Magenta
              [160, 32, 240, 255],   # purple
              [255, 255, 240, 255]]  # ivory
    shape_mesh = []
    for idx in range(n_parts):
        part_voxel = partsdf2voxel(np.asarray(points[idx:idx+1]), np.asarray(values[idx:idx+1]), vox_dim)
        vertices, triangles = libmcubes.marching_cubes(part_voxel, 0)
        if affine is not None:
            vertices = vertices * affine[idx, :, :1] + affine[idx, :, 1:] * vox_dim
        part_mesh = trimesh.Trimesh(vertices, triangles, face_colors=colors[idx % len(colors)])
        shape_mesh.append(part_mesh)
        # print(trimesh.visual.random_color())
    shape_mesh = trimesh.util.concatenate(shape_mesh)
    return shape_mesh