import os
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import torch
from torchvision import transforms


class ChairDataset(Dataset):
    def __init__(self, datadir, split = 'train'):
        positive_dir = datadir + '/chairs-data/positive/'
        negative_dir = datadir + '/chairs-data/negative/'

        isPositive = True
        data_size = 0

        imagesSide = []
        imagesTop = []
        imagesFront = []


        for i, foldername in enumerate([positive_dir, negative_dir]):

            data_files = sorted(os.listdir(foldername))

            class_size = len(data_files) // 3

            if split == 'train':
                data_files = data_files[:int(class_size * 0.8) * 3]
            else:
                data_files = data_files[int(class_size * 0.8) * 3:]


            class_size = len(data_files) // 3
            data_size += class_size
            for i, filename in enumerate(data_files):
                view = int(filename.split(".")[0])
                view = view % 3
                img = cv2.imread(foldername + filename)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.nan_to_num(img)

                if img is not None:
                    if view == 2:
                        imagesSide.append(1. - img / 255.)
                    elif view == 0:
                        imagesTop.append(1. - img / 255.)
                    else:
                        imagesFront.append(1. - img / 255.)
                else:
                    a = 0
            if isPositive:
                labels_positive = np.ones(class_size)
            else:
                labels_negative = np.zeros(class_size)
            isPositive = not isPositive




        #imagesFront = np.expand_dims(np.array(imagesFront), axis= 1)
        #imagesSide = np.expand_dims(np.array(imagesSide), axis= 1)
        #imagesTop = np.expand_dims(np.array(imagesTop), axis= 1)

        imagesFront = np.array(imagesFront)
        imagesSide = np.array(imagesSide)
        imagesTop = np.array(imagesTop)


        images = np.stack([imagesFront, imagesSide, imagesTop], axis=1)
        labels = np.concatenate([labels_positive,labels_negative])

        np.random.seed(234)
        np.random.shuffle(images)
        np.random.seed(234)
        np.random.shuffle(labels)

        self.images = images
        self.labels = labels


    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def __len__(self):
        return self.images.shape[0]


class EvaluateData(Dataset):
    def __init__(self, data_address):
        folder = data_address + '/evaluate-chairs/'

        imagesSide = []
        imagesTop = []
        imagesFront = []

        data_files = sorted(os.listdir(folder))
        for i, filename in enumerate(data_files):
            view = int(filename.split(".")[0])
            view = view % 3
            img = cv2.imread(folder + filename)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.nan_to_num(img)

            if img is not None:
                if view == 2:
                    imagesSide.append(1. - img / 255.)
                elif view == 0:
                    imagesTop.append(1. - img / 255.)
                else:
                    imagesFront.append(1. - img / 255.)

        imagesFront = np.array(imagesFront)
        imagesSide = np.array(imagesSide)
        imagesTop = np.array(imagesTop)

        self.images = np.stack([imagesFront, imagesSide, imagesTop], axis=1)
        self.labels = np.zeros((20,))
        self.labels[10:] += 1

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def __len__(self):
        return self.images.shape[0]


class SelfDataset(Dataset):
    def __init__(self, datadir, split='train'):
        positive_dir = datadir + '/goodimgs'
        negative_dir = datadir + '/badimgs/'

        isPositive = True
        data_size = 0

        imagesX = []
        imagesY = []
        imagesZ = []

        for i, foldername in enumerate([positive_dir, negative_dir]):

            data_files = sorted(os.listdir(foldername))

            class_size = len(data_files)

            if split == 'train':
                data_files = data_files[:int(class_size * 0.8)]
            else:
                data_files = data_files[int(class_size * 0.8):]

            class_size = len(data_files)
            data_size += class_size

            for i, f in enumerate(data_files):
                obj_path = os.path.join(foldername, f)
                image_x = 1 - cv2.imread(os.path.join(obj_path, 'x.jpeg'), cv2.IMREAD_UNCHANGED) / 255.
                image_y = 1 - cv2.imread(os.path.join(obj_path, 'y.jpeg'),  cv2.IMREAD_UNCHANGED) / 255.
                image_z = 1 - cv2.imread(os.path.join(obj_path, 'z.jpeg'),  cv2.IMREAD_UNCHANGED) / 255.

                image_x = cv2.resize(image_x, (224, 224))
                image_y = cv2.resize(image_y, (224, 224))
                image_z = cv2.resize(image_z, (224, 224))

                imagesX.append(image_x)
                imagesY.append(image_y)
                imagesZ.append(image_z)

            if isPositive:
                labels_positive = np.ones(class_size)
            else:
                labels_negative = np.zeros(class_size)
            isPositive = not isPositive
            a = 0

        imagesX = np.array(imagesX)
        imagesY = np.array(imagesY)
        imagesZ = np.array(imagesZ)

        images = np.stack([imagesX, imagesY, imagesZ], axis=1)
        labels = np.concatenate([labels_positive, labels_negative])

        np.random.seed(234)
        np.random.shuffle(images)
        np.random.seed(234)
        np.random.shuffle(labels)

        self.images = images
        self.labels = labels

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

    def __len__(self):
        return self.images.shape[0]

        a = 0


