from config import getConfig
from dataload import get_dataloader
from model_partae.trainAE import trainAE
from model_partae.testAE import testAE


def mainAE():
    # load config parameters here
    configMain = getConfig()

    configMain.mode = "test"
    # if training selected
    # here the autoencoder will be trained for each part
    if configMain.mode == "train":
        shuffle = False
        useAllPts = True
        trainData = get_dataloader(configMain.mode, configMain, useAllPts, shuffle)
        configMain.mode = "val"
        valData = get_dataloader(configMain.mode, configMain, useAllPts, shuffle)
        trainAE(configMain, trainData, valData)

    # if testing selected
    # here the encoder is used to generate latent code only, the decoder isnt used
    if configMain.mode == "test":
        shuffle = False
        useAllPts = True
        configMain.cont = True
        configMain.batchSize = 1
        testData = get_dataloader(configMain.mode, configMain, useAllPts, shuffle)
        testAE(configMain, testData)


if __name__ == '__main__':
    mainAE()
