from agents import getAgent
import h5py
import os


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


def testAE(config, testData):
    agent = getAgent("partae")
    #encode and decode the test files only
    agent = getAgent("partae", config)


    #save them as mesh
    config.saveDir = "result/partae"
    config.saveFormat = "mesh"
    save_output(0, 0, config.saveDir, config.saveFormat)

    return False
