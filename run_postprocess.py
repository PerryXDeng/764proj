import os

command = "python file.py"
os.system(command)

command = "python convert_h5_vox.py --src data/orgh5 --out data/step1h5 --n_parts 6"
os.system(command)

command = "python fill_part_solid.py --src data/step1h5 --out data/step2h5/Chair"
os.system(command)

command = "python rescale_part_vox.py --src data/step2h5/Chair"
os.system(command)

command = "python ../data/sample_points_from_voxel.py --src data/step2h5 --category Chair"
os.system(command)