import torch.nn as nn


#Encoder structure based on imnet
class EncoderIM(nn.Module):
    def __init__(self, nLayers, fDim=32, zDim=128):
        super(EncoderIM, self).__init__()
        model = []

        inChannels = 1
        outChannels = fDim

    def forward(self, x):
        out = self.model(x)
        out = out.view(out.shape[0], -1)
        return out

#Decoder Structure
class DecoderIM(nn.Module):


#IMNET based AE for training part based encoder
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


