from networks.partAE import IMNetPart
from networks.seq2seq import Seq2SeqAE
from dataload.data_utils import n_parts_map


def get_network(name, config):
    if name == "part_ae":
        net = IMNetPart(config.nLayersE, config.efDim, config.nLayersD, config.dfDim, config.zDim)
        return net
    elif name == 'seq2seq':
        partFeatSize = config.zDim + config.boxparamSize
        enInputSize = partFeatSize + n_parts_map(config.maxNumParts) + 1
        deInputSize = partFeatSize
        net = Seq2SeqAE(enInputSize, deInputSize, config.hiddenSize)
        return net
    else:
        raise ValueError


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
