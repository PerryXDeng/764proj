import os
from tqdm import tqdm
import numpy as np
import torch
import h5py
from dataload import get_dataloader
from agents.pqnet import MainNetwork


def reconstruct(config):
    """run reconstruction"""
    # create the whole framwork
    pqnet = MainNetwork(config)

    # create dataloader
    test_loader = get_dataloader('test', config)

    # output dest
    save_dir = os.path.join(config.exp_dir, "results/rec-ckpt-{}-{}-p{}".format(config.ckpt, config.format,
                                                                                int(config.by_part)))
    # ensure_dir(save_dir)

    # run testing
    pbar = tqdm(test_loader)
    for data in pbar:
        data_id = data['path'][0].split('/')[-1].split('.')[0]
        with torch.no_grad():
            pqnet.reconstruct(data)
            output_shape = pqnet.generate_shape(format=config.format, by_part=config.by_part)

        save_output(output_shape, data_id, save_dir, format=config.format)


def encode(config):
    """encode each data to shape latent space """
    # create the whole framwork
    pqnet = MainNetwork(config)

    # output dest
    save_dir = os.path.join(config.exp_dir, "results/enc-ckpt-{}".format(config.ckpt))
    # ensure_dir(save_dir)

    phases = ['train', 'val', 'test']
    for pha in phases:
        data_loader = get_dataloader(pha, config, shuffle=False)

        save_phase_dir = os.path.join(save_dir, pha)
        # ensure_dir(save_phase_dir)

        pbar = tqdm(data_loader)
        for data in pbar:
            data_id = data['path'][0].split('/')[-1].split('.')[0]
            with torch.no_grad():
                shape_code = pqnet.encode(data).detach().cpu().numpy()

                save_path = os.path.join(save_phase_dir, "{}.npy".format(data_id))
                np.save(save_path, shape_code)


def decode(config):
    """decode given latent codes to final shape"""
    # create the whole framwork
    pqnet = MainNetwork(config)

    # load source h5 file
    with h5py.File(config.fake_z_path, 'r') as fp:
        all_zs = fp['zs'][:]

    # output dest
    fake_name = config.fake_z_path.split('/')[-1].split('.')[0]
    save_dir = os.path.join(config.exp_dir, "results/{}-{}-p{}".format(fake_name, config.format,
                                                                       int(config.by_part)))
    # ensure_dir(save_dir)

    # decoding
    pbar = tqdm(range(all_zs.shape[0]))
    for i in pbar:
        z = all_zs[i]
        z1, z2 = np.split(z, 2)
        z = np.stack([z1, z2])
        z = torch.tensor(z, dtype=torch.float32).unsqueeze(1).cuda()
        with torch.no_grad():
            pqnet.decode_seq(z)
            output_shape = pqnet.generate_shape(format=config.format, by_part=config.by_part)

        data_id = "%04d" % i
        save_output(output_shape, data_id, save_dir, format=config.format)


def save_output(shape, data_id, save_dir, format):
    if format == 'voxel':
        save_path = os.path.join(save_dir, "{}.h5".format(data_id))
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset('voxel', data=shape, compression=9)
    elif format == "mesh":
        save_path = os.path.join(save_dir, "{}.obj".format(data_id))
        shape.export(save_path)
    else:
        raise NotImplementedError


def test(config, testData):
    # create experiment config
    if config.rec:
        reconstruct(config)
    elif config.enc:
        encode(config)
    elif config.dec:
        decode(config)
    else:
        pass




