import os
from evaluate_model import score_main
import torch

# you might want to change this to the director that contains mesh_to_image.py
os.chdir('/home/maham/PycharmProjects/Branch2/764proj')

outdir = "/home/maham/PycharmProjects/Branch2/764proj/results/goodimgs"
trained_model_dir = "/home/maham/Desktop/models"
meshdir = "/home/maham/PycharmProjects/Branch2/764proj/results/mix"
#
command = 'python3 mesh_to_image.py --dir {}'.format(meshdir)
os.system(command)

scorelist = []

import operator
from shutil import copyfile
import shutil

if not os.path.isdir("results/final"):
    os.mkdir("results/final")

def score_multiple_models(outdir):
    device = torch.device('cuda:0')
    image_files = os.listdir(outdir)
    image_files = sorted(image_files)
    for f in image_files:
        print('model:', f)
        s = score_main(os.path.join(outdir, f), trained_model_dir, device)
        sVal = s.cpu().detach().numpy()
        sVal = sVal.item()
        scorelist.append([f, sVal])
    scorelist.sort(key=operator.itemgetter(1))
    idx = 1
    print(scorelist)
    for k in scorelist:
        f = k[0]
        src = "results/mix/model_"+f+".obj"
        dst = "results/final/" + "model_" + str(idx) + ".obj"
        shutil.copy(src, dst)
        idx = idx + 1


score_multiple_models(outdir)
