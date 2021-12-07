import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from utils import drawPartsBBOXVoxel
from networks import get_network, set_requires_grad
from tensorboardX import SummaryWriter
from utils import TrainClock


class Seq2SeqAgent(object):
    def __init__(self, config):
        super(Seq2SeqAgent, self).__init__()
        self.logDir = config.logDir
        self.stop_weight = config.stop_weight
        self.boxparam_size = config.boxparam_size
        self.teacher_decay = config.teacher_decay
        self.teacher_forcing_ratio = 0.5
        self.net = self.buildNet(config)
        self.bce_min = torch.tensor(1e-3, dtype=torch.float32, requires_grad=True).cuda()
        self.clock = TrainClock()
        self.chkpDir = config.chkpDir

        self.optimizer = optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

        self.rec_criterion = nn.MSELoss(reduction='none').cuda()
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()

        self.teacher_forcing_ratio *= self.teacher_decay

        # set tensorboard writers
        self.trainTB = SummaryWriter(os.path.join(self.logDir, 'train.events'))
        self.valTB = SummaryWriter(os.path.join(self.logDir, 'val.events'))

    def buildNet(self, config):
        # restore part encoder
        part_imnet = get_network('part_ae', config)
        if not os.path.exists(config.partae_modelpath):
            raise ValueError("Pre-trained part_ae path not exists: {}".format(config.partae_modelpath))
        part_imnet.load_state_dict(torch.load(config.partae_modelpath)['model_state_dict'])
        print("Load pre-trained part AE from: {}".format(config.partae_modelpath))
        self.part_encoder = part_imnet.encoder.cuda().eval()
        self.part_decoder = part_imnet.decoder.cuda().eval()
        set_requires_grad(self.part_encoder, requires_grad=False)
        set_requires_grad(self.part_decoder, requires_grad=False)
        del part_imnet
        # build rnn
        net = get_network('seq2seq', config).cuda()
        return net

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

    def updateLearningRate(self):
        """record and update learning rate"""
        self.trainTB.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if self.clock.epoch < 2000:
            self.scheduler.step(self.clock.epoch)

    # validation for one step
    def valFunc(self, data):
        self.net.eval()

        #   dont update gradients in this step
        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.recordLosses(losses, 'validation')

        return outputs, losses

    def forward(self, data):
        row_vox3d = data['vox3d']  # (B, max_n_parts, 1, vox_dim, vox_dim, vox_dim)
        part_labels = data['part_labels_onehot'] # (B, max_n_parts, total_num_parts)
        batch_size, max_n_parts, vox_dim = row_vox3d.size(0), row_vox3d.size(1), row_vox3d.size(-1)
        batch_n_parts = data['n_parts']
        target_stop = data['sign'].cuda()
        bce_mask = data['mask'].cuda()
        affine_input = data['affine_input'].cuda()
        affine_target = data['affine_target'].cuda()
        cond = data['cond'].cuda()

        batch_vox3d = row_vox3d.view(-1, 1, vox_dim, vox_dim, vox_dim).cuda()
        total_n_parts = part_labels.shape[-1]
        batch_labels = part_labels.view(-1, total_n_parts).cuda() # (B * max_n_parts, total_n_parts)
        with torch.no_grad():
            part_geo_features = self.part_encoder(batch_vox3d)  # (B * max_n_parts, z_dim)
            part_geo_features = part_geo_features.view(batch_size, max_n_parts, -1).transpose(0, 1)
            cond_pack = cond.unsqueeze(0).repeat(affine_input.size(0), 1, 1)
            print("part geo", part_geo_features.shape)
            print("affine", affine_input.shape)
            print("cond pack", cond_pack.shape)
            print("labels", batch_labels.shape)

            target_part_geo = part_geo_features.detach()
            part_feature_seq = torch.cat([part_geo_features, affine_input, cond_pack], dim=2)
            # part_feature_seq = torch.cat([part_geo_features, affine_input, cond_pack, batch_labels], dim=2)
            print("feature seq catted", part_feature_seq.shape)
            part_feature_seq = pack_padded_sequence(part_feature_seq, batch_n_parts, enforce_sorted=False)
            _, seq_lengths = pad_packed_sequence(part_feature_seq)  # self to self translation
            target_seq = torch.cat([target_part_geo, affine_target], dim=2)
            # target_seq = torch.cat([target_part_geo, affine_target, batch_labels.detach()], dim=2)

        output_seq, output_stop = self.net(part_feature_seq, target_seq, self.teacher_forcing_ratio)

        bce_loss = self.bce_criterion(output_stop, target_stop) * bce_mask * self.stop_weight

        code_rec_loss = self.rec_criterion(output_seq[:, :, :-self.boxparam_size], target_part_geo) * bce_mask
        param_rec_loss = self.rec_criterion(output_seq[:, :, -self.boxparam_size:], affine_target) * bce_mask

        code_rec_loss = torch.sum(code_rec_loss) / (torch.sum(bce_mask) * code_rec_loss.size(2))
        param_rec_loss = torch.sum(param_rec_loss) / (torch.sum(bce_mask) * param_rec_loss.size(2))
        bce_loss = torch.max(torch.sum(bce_loss) / torch.sum(bce_mask), self.bce_min)

        return output_seq, {"code": code_rec_loss, "param": param_rec_loss, "stop": bce_loss}

    def visualize_batch(self, data, mode, outputs=None, **kwargs):
        tb = self.trainTB if mode == 'train' else self.valTB

        n_parts = data['n_parts'][0]
        affine_input = data['affine_input'][:n_parts, 0].detach().cpu().numpy()
        affine_target = data['affine_target'][:n_parts, 0].detach().cpu().numpy()
        affine_output = outputs[:n_parts, 0, -self.boxparam_size:].detach().cpu().numpy()

        # draw box
        bbox_proj = drawPartsBBOXVoxel(affine_input)
        tb.add_image("bbox_input", torch.from_numpy(bbox_proj), self.clock.step, dataformats='HW')
        bbox_proj = drawPartsBBOXVoxel(affine_target)
        tb.add_image("bbox_target", torch.from_numpy(bbox_proj), self.clock.step, dataformats='HW')
        bbox_proj = drawPartsBBOXVoxel(affine_output)
        tb.add_image("bbox_output", torch.from_numpy(bbox_proj), self.clock.step, dataformats='HW')

    def update_teacher_forcing_ratio(self):
        self.teacher_forcing_ratio *= self.teacher_decay

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