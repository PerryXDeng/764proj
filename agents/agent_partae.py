import torch.nn as nn
from utils import visualizeSDF, projVoxelXYZ
import torch
from utils import TrainClock
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
from utils import buildNet


# Agent class for Part based Auto Encoder
class AgentPartAE(object):
    def __init__(self, config):
        # get the relevant information from config
        self.logDir = config.logDir
        self.modelDir = config.modelDir
        self.clock = TrainClock()
        self.batchSize = config.batchSize
        self.ptsBatchSize = config.ptsBatchSize
        self.resolution = config.resolution
        self.chkpDir = config.chkpDir
        # set the loss function as MSE
        self.criterion = nn.MSELoss().cuda()

        # get the relevant network
        self.net = buildNet('partae', config)

        # set ADAM optimizer
        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lrStep)

        # set tensorboard writers
        self.trainTB = SummaryWriter(os.path.join(self.logDir, 'train.events'))
        self.valTB = SummaryWriter(os.path.join(self.logDir, 'val.events'))

    # define the forward function
    def forward(self, data):
        # get the voxel data, the corresponding points and their values
        inVox3d = data['vox3d'].cuda()  # (shape_batch_size, 1, dim, dim, dim)
        points = data['points'].cuda()  # (shape_batch_size, points_batch_size, 3)
        targetSDF = data['values'].cuda()  # (shape_batch_size, points_batch_size, 1)

        # outSDF is the output of decoder from IMNET
        outSDF = self.net(points, inVox3d)
        # get the loss between outSDF and targetSDF
        loss = self.criterion(outSDF, targetSDF)

        return outSDF, {"mse": loss}

    # back propogation
    def updateNetwork(self, losses):
        loss = sum(losses.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # keep a record of the losses
    def recordLosses(self, losses, mode='train'):
        lossVals = {k: v.item() for k, v in losses.items()}

        # record loss to tensorboard
        tb = self.trainTB if mode == 'train' else self.valTB
        for k, v in lossVals.items():
            tb.add_scalar(k, v, self.clock.step)

    # define the train function for one step
    def trainFunc(self, data):
        self.net.train()
        outputs, losses = self.forward(data)

        self.updateNetwork(losses)
        self.recordLosses(losses, 'train')

        return outputs, losses

    # validation for one step
    def valFunc(self, data):
        self.net.eval()

        #   dont update gradients in this step
        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.recordLosses(losses, 'validation')

        return outputs, losses

    # record and update learning rate
    def updateLearningRate(self):
        self.trainTB.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step(self.clock.epoch)

    # visualizing current batch and storing output
    def visualizeCurBatch(self, data, mode, outputs=None):
        tb = self.trainTB if mode == 'train' else self.valTB

        # get voxel, points and their values (sdf) from data
        partsVoxel = data['vox3d'][0][0].numpy()
        dataPts64 = data['points'][0].numpy() * self.resolution
        dataVals64 = data['values'][0].numpy()
        # sdf value from the part ae net
        outSDF = outputs[0].detach().cpu().numpy()

        target = visualizeSDF(dataPts64, dataVals64, concat=True, voxDim=self.resolution)
        output = visualizeSDF(dataPts64, outSDF, concat=True, voxDim=self.resolution)
        voxProj = projVoxelXYZ(partsVoxel, concat=True)
        tb.add_image("voxel", torch.from_numpy(voxProj), self.clock.step, dataformats='HW')
        tb.add_image("target", torch.from_numpy(target), self.clock.step, dataformats='HW')
        tb.add_image("output", torch.from_numpy(output), self.clock.step, dataformats='HW')

    # -- Check Points --- #

    # Saving checkpoint
    def saveChkPt(self, pathIn=None):

        if pathIn is None:
            savePath = os.path.join(self.chkpDir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Checkpoint saved at {}".format(savePath))
        else:
            savePath = os.path.join(self.chkpDir, "{}.pth".format(pathIn))

        if isinstance(self.net, nn.DataParallel):
            torch.save({
                'clock': self.clock.makeChkPt(),
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, savePath)
        else:
            torch.save({
                'clock': self.clock.makeChkPt(),
                'model_state_dict': self.net.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, savePath)

        self.net.cuda()

    # loading Check Point
    def loadChkPt(self, pathIn=None):
        pathIn = pathIn if pathIn == 'latest' else "ckpt_epoch{}".format(pathIn)
        load_path = os.path.join(self.chkpDir, "{}.pth".format(pathIn))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Checkpoint loaded from {}".format(load_path))

        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restoreChkPt(checkpoint['clock'])


