from dataload import get_dataloader
from config import getConfig
from utils import cycle
from model_main.trainMain import train
from model_main.testMain import test

def main():
    config = getConfig()
    config.module = "seq2seq"

    # create dataloader
    config.mode = "train"
    config.byPart = True

    # if training selected
    # here the autoencoder will be trained for each part
    if config.mode == "train":
        train_loader = get_dataloader('train', config)
        val_loader = get_dataloader('val', config)
        val_loader = cycle(val_loader)
        train(config, train_loader, val_loader)

    # if testing selected
    # here the encoder is used to generate latent code only, the decoder isnt used
    if config.mode == "test":
        config.cont = True
        config.batchSize = 1
        config.rec = True
        config.enc = False
        config.dec = False

        testData = get_dataloader(config.mode, config)
        test(config, testData)





if __name__ == '__main__':
    main()