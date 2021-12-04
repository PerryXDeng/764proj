from networks.partAE import IMNetPart
from dataload.data_utils import n_parts_map


def get_network(name, config):
    net = IMNetPart(config.nLayersE, config.efDim, config.nLayersD, config.dfDim, config.zDim)
    return net

