import os
import cv2

from new_model import Classifer, Single_Classifier

import torch
import torch.nn.functional as F


def score_single_chair_single_model(chair_dir, model_path, device):
    img_side  = 1 - cv2.imread(os.path.join(chair_dir, 'x.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
    img_side = cv2.resize(img_side, (224, 224))

    img_top = 1 - cv2.imread(os.path.join(chair_dir, 'y.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
    img_top = cv2.resize(img_top, (224, 224))

    img_front = 1 - cv2.imread(os.path.join(chair_dir, 'z.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
    img_front = cv2.resize(img_front, (224, 224))

    imageSide = torch.tensor(img_side).reshape(1, 1, 224, 224).to(device).float()
    imageTop = torch.tensor(img_top).reshape(1, 1, 224, 224).to(device).float()
    imageFront = torch.tensor(img_front).reshape(1, 1, 224, 224).to(device).float()

    network_side = Single_Classifier().to(device)
    network_side.eval()
    sd = torch.load(os.path.join(model_path, 'side.pt'))['model_state_dict']
    network_side.load_state_dict(sd)

    network_top = Single_Classifier().to(device)
    network_top.eval()
    sd = torch.load(os.path.join(model_path, 'top.pt'))['model_state_dict']
    network_top.load_state_dict(sd)

    network_front = Single_Classifier().to(device)
    network_front.eval()
    sd = torch.load(os.path.join(model_path, 'front.pt'))['model_state_dict']
    network_front.load_state_dict(sd)

    o_side = network_side(imageSide)
    score_side = F.softmax(o_side, dim = 1)[0, 1]

    o_top = network_top(imageTop)
    score_top = F.softmax(o_top, dim = 1)[0, 1]

    o_front = network_front(imageFront)
    score_front = F.softmax(o_front, dim = 1)[0, 1]

    score = min([score_side, score_front, score_top])
    return score


def score_single_chair(chair_dir, model_path,device):
    net = Classifer().to(device)
    net.eval()
    sd = torch.load(os.path.join(model_path, 'checkpoint_shared.pt'))['model_state_dict']
    net.load_state_dict(sd)

    img_side = 1 - cv2.imread(os.path.join(chair_dir, 'x.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
    img_side = cv2.resize(img_side, (224, 224))

    img_top = 1 - cv2.imread(os.path.join(chair_dir, 'y.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
    img_top = cv2.resize(img_top, (224, 224))

    img_front = 1 - cv2.imread(os.path.join(chair_dir, 'z.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
    img_front = cv2.resize(img_front, (224, 224))

    imageSide = torch.tensor(img_side).reshape(1, 1, 224, 224).to(device).float()
    imageTop = torch.tensor(img_top).reshape(1, 1, 224, 224).to(device).float()
    imageFront = torch.tensor(img_front).reshape(1, 1, 224, 224).to(device).float()

    o = net(imageTop, imageSide, imageFront)
    score = F.softmax(o, dim = 1)[0, 1]
    return score

def score_main(chair_dir, model_path, device):
    score_shared = score_single_chair(chair_dir, model_path,device)
    score_sep = score_single_chair_single_model(chair_dir,model_path, device)
    alpha = 2
    score_general =  (score_shared + alpha * score_sep) / (alpha + 1)

    print('score minimum model: ', score_sep.item(), 'score shared model:', score_shared.item(), 'overall score:', score_general.item())
    return score_general


