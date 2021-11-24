from networks import getNetwork
import torch.nn as nn
from agents.base import BaseAgent
from utils import visualizeSDF, projVoxelXYZ
import torch


class AgentPartAE(BaseAgent):
    def __init__(self, config):
        super(AgentPartAE, self).__init__(config)
        self.ptsBatchSize = config.ptsBatchSize
        self.resolution = config.resolution
        self.batchSize = config.batchSize
        self.criterion = nn.MSELoss().cuda()

    def buildNet(self, config):
        net = getNetwork('partae', config)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def forward(self, data):
        input_vox3d = data['vox3d'].cuda()  # (shape_batch_size, 1, dim, dim, dim)
        points = data['points'].cuda()  # (shape_batch_size, points_batch_size, 3)
        target_sdf = data['values'].cuda()  # (shape_batch_size, points_batch_size, 1)

        output_sdf = self.net(points, input_vox3d)

        loss = self.criterion(output_sdf, target_sdf)
        return output_sdf, {"mse": loss}

    def visualizeCurBatch(self, data, mode, outputs=None):
        tb = self.train_tb if mode == 'train' else self.val_tb

        parts_voxel = data['vox3d'][0][0].numpy()
        data_points64 = data['points'][0].numpy() * self.resolution
        data_values64 = data['values'][0].numpy()
        output_sdf = outputs[0].detach().cpu().numpy()

        target = visualizeSDF(data_points64, data_values64, concat=True, voxDim=self.resolution)
        output = visualizeSDF(data_points64, output_sdf, concat=True, voxDim=self.resolution)
        voxel_proj = projVoxelXYZ(parts_voxel, concat=True)
        tb.add_image("voxel", torch.from_numpy(voxel_proj), self.clock.step, dataformats='HW')
        tb.add_image("target", torch.from_numpy(target), self.clock.step, dataformats='HW')
        tb.add_image("output", torch.from_numpy(output), self.clock.step, dataformats='HW')
