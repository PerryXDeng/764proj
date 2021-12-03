from agents import get_agent
import numpy as np
from utils import cycle, sdf2voxel, voxel2mesh
import trimesh
import os
import torch
import h5py
import mcubes as libmcubes


def testAffine(agent, testData):
    path = "/localhome/mta122/PycharmProjects/764proj/data/Chair/172.h5"
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        parts_voxel = data_dict['parts_voxel_scaled64'][:].astype(np.float)
        data_points64 = data_dict['points_64'][:]
        data_values64 = data_dict['values_64'][:]
        translation = data_dict['translations'][:]
        size = data_dict['size'][:]
        affine = np.concatenate([translation, size], axis=1)
        batch_affine = torch.tensor(affine, dtype=torch.float32)  # (n_parts, 6)

    idx = 0
    part_trans = affine[idx, :1]
    # B = batch_affine[idx, :, 1:]
    print(part_trans)
    print(batch_affine)


# output all parts of output chair
def outputWholeChairDIRECT(agent, config, testData):
    config.ckpt = 'latest'
    # load check point
    colors = [[0, 0, 255, 255],  # blue
              [0, 255, 0, 255],  # green
              [255, 0, 0, 255],  # red
              [255, 255, 0, 255],  # yellow
              [0, 255, 255, 255],  # cyan
              [255, 0, 255, 255],  # Magenta
              [160, 32, 240, 255],  # purple
              [255, 255, 240, 255]]  # ivory


    # scene = trimesh.Scene()
    filenum = 38037
    voxDim = 64
    path = "/localhome/mta122/PycharmProjects/764proj/data/Chair/{}.h5".format(filenum)
    with h5py.File(path, 'r') as obj_hf:
        n_parts = obj_hf.attrs['n_parts']
        parts_voxel_scaled64 = np.array(obj_hf['parts_voxel_scaled64'])
        parts_translations = np.array(obj_hf['translations'])
        parts_scales = np.array(obj_hf['scales'])
        shape_mesh = []
        for i in range(n_parts):
            vertices, triangles = libmcubes.marching_cubes(parts_voxel_scaled64[i,:], 0)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.apply_translation((-32, -32, -32))
            mesh.apply_scale([parts_scales[i], parts_scales[i], parts_scales[i]])
            mesh.apply_translation(parts_translations[i,:])
            shape_mesh.append(mesh)

    shape_mesh = trimesh.util.concatenate(shape_mesh)
    savePath = os.path.join(config.saveDir, "model_{}.obj".format(filenum))
    shape_mesh.export(savePath, file_type='obj')


# output all parts of output chair
def outputWholeChair(agent, config, testData):
    config.ckpt = 'latest'
    # load check point
    colors = [[0, 0, 255, 255],  # blue
              [0, 255, 0, 255],  # green
              [255, 0, 0, 255],  # red
              [255, 255, 0, 255],  # yellow
              [0, 255, 255, 255],  # cyan
              [255, 0, 255, 255],  # Magenta
              [160, 32, 240, 255],  # purple
              [255, 255, 240, 255]]  # ivory

    if config.cont:
        agent.loadChkPt(config.ckpt)

        testData = cycle(testData)
        data = next(testData)
        numShapes = 1

        for j in range(numShapes):
            nParts = data['n_parts']  # (shape_batch_size, 1)
            shape_mesh = []
            for i in range(nParts):
                voxDim = 64

                points = data['points'][0].numpy() * voxDim
                translation = data['translations'][0].numpy().flatten()
                size = data['size'][0].numpy().flatten()
                scale = data['scales'][0].numpy().flatten()
                affine = np.concatenate([translation, size])
                outputs, losses = agent.valFunc(data)
                values = outputs[0].detach().cpu().numpy()
                targetSDF = data['values'][0].numpy()

                voxels = sdf2voxel(points, targetSDF, voxDim=voxDim)
                mesh = voxel2mesh(voxels, config, export=False, name="", affine=affine, voxDim=voxDim, idx=i,
                                  size=size, translation=translation,
                                  inCol=colors[i % len(colors)],scale=scale)

                shape_mesh.append(mesh)
                # voxProj, targetImg, outputImg = agent.visualizeCurBatch(data, 'val')
                data = next(testData)

            shape_mesh = trimesh.util.concatenate(shape_mesh)
            savePath = os.path.join(config.saveDir, "model_{}.obj".format(j))
            shape_mesh.export(savePath, file_type='obj')


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

            print(np.shape(totalCode))

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
    agent = get_agent("partae", config)

    # save them as mesh
    config.saveDir = "results/partae"
    config.saveFormat = "mesh"

    # reconstruct(agent, config, testData)
    # makeCode(agent, config, testData)
    outputWholeChair(agent, config, testData)
    # outputRandomPart(agent, config, testData)
    # testAffine(agent, testData)

    return False
