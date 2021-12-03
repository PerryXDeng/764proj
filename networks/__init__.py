from networks.partAE import IMNetPart


def get_network(name, config):
    if name == "partae":
        net = IMNetPart(config.nLayersE, config.efDim, config.nLayersD, config.dfDim, config.zDim)
    return net
