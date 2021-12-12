

import os

import trimesh
from utils import projVoxelXYZ
import h5py
import numpy as np
from PIL import Image


from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, default='' )

args = parser.parse_args()


base_dir = args.dir

#meshdir = "/localhome/ama240/Desktop/764proj/results/mix/test"
meshdir = os.path.join(base_dir, 'meshes')
good = True
outdir = os.path.join(base_dir, 'results')
#outdir = "/localhome/ama240/Desktop/764proj/results"
objs = os.listdir(meshdir)

#########################################################
############# MAKE THE INITAL VOXELIZE ##################
#########################################################

idx = 1
# scale = 64
for obj in objs:
    d_path = os.path.join(meshdir, obj)
    if obj.startswith('model'):
        if good:
            s_path = os.path.join(outdir, "tempgood")
        else:
            s_path = os.path.join(outdir, "tempbad")
        mesh = trimesh.load(d_path)
        if not os.path.exists(s_path):
          os.makedirs(s_path)
        e_path = os.path.join(s_path, str(idx)+".off")
        idx = idx + 1
        mesh.export(e_path)

dim = 64
#if good:
#    data_dir = "/localhome/ama240/Desktop/764proj/results/tempgood"
#else:
#    data_dir = "/localhome/ama240/Desktop/764proj/results/tempbad"

data_dir = s_path

save_dir = os.path.join(outdir, 'tempvoxels')
objs = os.listdir(data_dir)
for obj in objs:
    d_path = os.path.join(data_dir, obj)
    s_path = os.path.join(save_dir, obj.replace(".off",""))
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    e_path = os.path.join(s_path, "object_all.h5")
    command = './voxelize occ {} {} --height {} --width {} --depth {}'.format(
        d_path, e_path, dim, dim, dim)
    os.system(command)

#########################################################
################ CONDITION ##############################
#########################################################

#command = 'python3 convert_h5_vox.py --src /localhome/ama240/Desktop/764proj/results/tempvoxels --out /localhome/ama240/Desktop/764proj/results/tempvoxels2 --n_parts 1'
command = 'python3 convert_h5_vox.py --src {} --out {} --n_parts 1'.format(save_dir, os.path.join(outdir, 'tempvoxels2'))

os.system(command)

#command = "python3 fill_part_solid.py --src /localhome/ama240/Desktop/764proj/results/tempvoxels2 --out /localhome/ama240/Desktop/764proj/results/tempvoxels3/Chair"
command = "python3 fill_part_solid.py --src {} --out {}".format(os.path.join(outdir, 'tempvoxels2'), os.path.join(outdir, 'tempvoxels3/Chair'))
os.system(command)

#command = "python3 rescale_part_vox.py --src /localhome/ama240/Desktop/764proj/results/tempvoxels3/Chair"
command = "python3 rescale_part_vox.py --src {}".format(os.path.join(outdir, 'tempvoxels3/Chair'))
os.system(command)

command = "python3 /localhome/ama240/Desktop/764proj/data/sample_points_from_voxel.py --src {} --category Chair".format(os.path.join(outdir, 'tempvoxels3'))
os.system(command)


#########################################################
################ SAVE THE IMAGE #########################
#########################################################
#
#final_vox_dir = "/localhome/ama240/Desktop/764proj/results/tempvoxels3/Chair"
final_vox_dir = os.path.join(outdir, 'tempvoxels3/Chair')
voxelModels = os.listdir(final_vox_dir)
if good:
    #image_dir = "/localhome/ama240/Desktop/764proj/results/goodimgs"
    image_dir = os.path.join(outdir, 'imgs')
else:
    image_dir = "/localhome/ama240/Desktop/764proj/results/badimgs"

for vox in voxelModels:
    filename = os.path.join(final_vox_dir, vox)
    with h5py.File(filename, "r") as data_dict:
        voxels = data_dict['parts_voxel_scaled64'][0].astype(np.float)
    img1, img2, img3 = projVoxelXYZ(voxels)
    main_path = os.path.join(image_dir, vox.replace(".h5", ""))
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    # top
    img1 = 255 - img1
    im = Image.fromarray(img1)
    path = os.path.join(main_path, "x.jpeg")
    im = im.resize((256, 256), Image.ANTIALIAS)
    im = im.rotate(180)
    im.save(path)

    # bottom
    img2 = 255 - img2
    im = Image.fromarray(img2)
    im = im.resize((256, 256), Image.ANTIALIAS)
    im = im.rotate(90)
    path = os.path.join(main_path, "y.jpeg")
    im.save(path)

    # side
    img3 = 255 - img3
    im = Image.fromarray(img3)
    im = im.resize((256, 256), Image.ANTIALIAS)
    im = im.rotate(90)
    path = os.path.join(main_path, "z.jpeg")
    im.save(path)
