import os
import trimesh
import h5py
import numpy as np

dim = 64
data_dir = "data/Chairs"
save_dir = "data/off"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


part_objs = os.listdir(data_dir)
scale = 64
idx = 1
for part_obj in part_objs:
    idx = 1
    d_path = os.path.join(data_dir, part_obj)
    s_path = os.path.join(save_dir, part_obj)
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    total_files = os.listdir(d_path)
    total_files.sort()
    for file in total_files:
        mesh = trimesh.load(os.path.join(d_path, file))
        mesh.apply_scale([scale, scale, scale])
        mesh.apply_translation((32, 32, 32))
        name = str(idx) + ".off"
        idx = idx + 1
        e_path = os.path.join(s_path, name)
        mesh.export(e_path)

#./voxelize occ input/ out.h5 --height 64 --width 64 --depth 64'
data_dir = "off"
save_dir = "orgh5"
for part_obj in part_objs:
    d_path = os.path.join(data_dir, part_obj)
    s_path = os.path.join(save_dir, part_obj)
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    e_path = os.path.join(save_dir, part_obj, "object_all.h5")
    command = './voxelize occ {} {} --height {} --width {} --depth {}'.format(
        d_path, e_path, dim, dim, dim)
    os.system(command)

