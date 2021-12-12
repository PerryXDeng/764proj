import torch
import torchvision
import torch.nn.functional as F
from torch import nn



class ResnetModel(nn.Module):
    def __init__(self, pretrained = False, backbone = 'resnet34', num_layers = 5):
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(
            pretrained = pretrained
        )

        self.num_layers = num_layers

        self.dim_control = nn.Conv2d(in_channels= 1, out_channels=3, kernel_size=3, padding=1)
    def forward(self, x):
        B, C, H, W = x.shape

        if C == 1:
            x = self.dim_control(x)

        #x = self.model(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        if self.num_layers > 1:
            x = self.model.layer1(x)
        if self.num_layers > 2:
            x = self.model.layer2(x)
        if self.num_layers > 3:
            x = self.model.layer3(x)
        if self.num_layers > 4:
            x = self.model.layer4(x)

        x = self.model.avgpool(x)

        x = nn.Flatten()(x)

        return x


class Classifer(nn.Module):

    def __init__(self):
        super().__init__()

        self.top_net = ResnetModel()
        self.front_net = ResnetModel()
        self.side_net = ResnetModel()

        self.fc1 = nn.Linear(in_features=3 * 512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x_top, x_front, x_side):
        feature_top = self.top_net(x_top)
        feature_front = self.front_net(x_front)
        feature_side = self.side_net(x_side)

        feature = torch.cat([feature_top, feature_front, feature_side], dim=1)

        o = self.fc1(feature)
        o = self.bn1(o)
        o = self.relu(o)

        o = F.dropout(o, p=0.5, training=self.training)

        o = self.fc2(o)
        o = self.bn2(o)
        o = self.relu(o)

        o = F.dropout(o, p=0.3, training=self.training)

        o = self.fc3(o)

        return o



class Shared_Classifer(nn.Module):

    def __init__(self):
        super().__init__()


        self.resnet_model = ResnetModel()

        self.fc1 = nn.Linear(in_features=3 * 512, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=2)

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)


    def forward(self, x_top, x_front, x_side):
        feature_top = self.resnet_model(x_top)
        feature_front = self.resnet_model(x_front)
        feature_side = self.resnet_model(x_side)

        feature = torch.cat([feature_top, feature_front, feature_side], dim = 1)

        o = self.fc1(feature)
        o = self.bn1(o)
        o = self.relu(o)

        o = F.dropout(o, p = 0.5, training = self.training)

        o = self.fc2(o)
        o = self.bn2(o)
        o = self.relu(o)

        o = F.dropout(o, p=0.3, training=self.training)

        o = self.fc3(o)


        return o

class Single_Classifier(nn.Module):
    def __init__(self):
        super().__init__()


        self.resnet_model = ResnetModel()

        self.fc1 = nn.Linear(in_features= 512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, image):
        feature = self.resnet_model(image)

        o = self.fc1(feature)
        o = self.bn1(o)
        o = self.relu(o)

        o = F.dropout(o, p=0.5, training=self.training)

        o = self.fc2(o)
        o = self.bn2(o)
        o = self.relu(o)

        o = F.dropout(o, p=0.3, training=self.training)

        o = self.fc3(o)

        return o

class Baseline_LeNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding='same')
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=56 * 56 * 64, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = nn.Flatten()(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = F.dropout(x, p = 0.4, training=self.training)

        x = self.fc2(x)

        return x


