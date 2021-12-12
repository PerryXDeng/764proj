import os
from evaluate_model import score_main
import torch

#you might want to change this to the director that contains mesh_to_image.py
os.chdir('/localhome/ama240/Desktop/new_code/geom_proj')

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dir", type=str, default='' )

args = parser.parse_args()

meshdir = args.dir
outdir = os.path.join(meshdir, 'results/imgs')
trained_model_dir = '/localhome/ama240/Desktop/saved_models'

command = 'python3 mesh_to_image.py --dir {}'.format(meshdir)
if not os.path.exists(outdir):
    os.system(command)


def score_multiple_models(data_dir):
    device = torch.device('cuda:0')
    image_files = os.listdir(data_dir)
    for f in image_files:
        print('image:', f)

        s = score_main(os.path.join(data_dir, f), trained_model_dir, device)




score_multiple_models(outdir)