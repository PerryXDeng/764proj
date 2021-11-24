import torch.nn as nn
import torch


# code taken from pqnet, will adjust as we go ahead

# Encoder structure based on imnet
class EncoderIM(nn.Module):
    def __init__(self, nLayers, fDim=32, zDim=128):
        super(EncoderIM, self).__init__()
        model = []

        inChannels = 1
        outChannels = fDim
        for i in range(nLayers - 1):
            model.append(nn.Conv3d(inChannels, outChannels, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=1,
                                   bias=False))  # FIXME: in IM-NET implementation, they use bias before BN
            model.append(nn.BatchNorm3d(num_features=outChannels, momentum=0.1))  # FIXME: momentum value
            model.append(nn.LeakyReLU(0.02))

            in_channels = outChannels
            outChannels *= 2

        model.append(nn.Conv3d(outChannels // 2, zDim, kernel_size=(4, 4, 4), stride=(1, 1, 1), padding=0))
        model.append(nn.Sigmoid())

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out


# Decoder Structure
class DecoderIM(nn.Module):
    def __init__(self, nLayers, fDim, zDim):
        """With skip connection"""
        super(DecoderIM, self).__init__()
        in_channels = zDim + 3
        out_channels = fDim * (2 ** (nLayers - 2))
        model = []
        for i in range(nLayers - 1):
            if i > 0:
                in_channels += zDim + 3
            if i < 4:
                model.append([nn.Linear(in_channels, out_channels), nn.Dropout(p=0.4), nn.LeakyReLU()])
            else:
                model.append([nn.Linear(in_channels, out_channels), nn.LeakyReLU()])
            in_channels = out_channels
            out_channels = out_channels // 2
        model.append([nn.Linear(in_channels, 1), nn.Sigmoid()])

        self.layer1 = nn.Sequential(*model[0])
        self.layer2 = nn.Sequential(*model[1])
        self.layer3 = nn.Sequential(*model[2])
        self.layer4 = nn.Sequential(*model[3])
        self.layer5 = nn.Sequential(*model[4])
        self.layer6 = nn.Sequential(*model[5])

    def forward(self, points, z):
        z_cat = torch.cat([points, z], dim=1)
        out = self.layer1(z_cat)
        out = self.layer2(torch.cat([out, z_cat], dim=1))
        out = self.layer3(torch.cat([out, z_cat], dim=1))
        out = self.layer4(torch.cat([out, z_cat], dim=1))
        out = self.layer5(torch.cat([out, z_cat], dim=1))
        out = self.layer6(out)
        return out


# IMNET based AE for training part based encoder
class IMNetPart(nn.Module):
    def __init__(self, nLayersE, efDim, nLayersD, dfDim, zDim):
        super(IMNetPart, self).__init__()
        self.encoder = EncoderIM(nLayersE, efDim, zDim)
        assert nLayersD == 6
        self.decoder = DecoderIM(nLayersD, dfDim, zDim)

    def forward(self, points, vox3d):
        shape_batch_size = points.size(0)
        point_batch_size = points.size(1)
        z = self.encoder(vox3d)  # (shape_batch_size, z_dim)
        # z = torch.cat([z, affine], dim=1)
        batch_z = z.unsqueeze(1).repeat((1, point_batch_size, 1)).view(-1, z.size(1))
        batch_points = points.view(-1, 3)

        out = self.decoder(batch_points, batch_z)
        out = out.view((shape_batch_size, point_batch_size, -1))
        return out
