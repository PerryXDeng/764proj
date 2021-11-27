from config import getConfig
from dataload import getDataLoader
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
        trainData = getDataLoader(configMain.mode, configMain, useAllPts, shuffle)
        configMain.mode = "val"
        valData = getDataLoader(configMain.mode, configMain, useAllPts, shuffle)
        trainAE(configMain, trainData, valData)

    # if testing selected
    # here the encoder is used to generate latent code only, the decoder isnt used
    if configMain.mode == "test":
        shuffle = False
        useAllPts = True
        configMain.cont = True
        configMain.batchSize = 1
        testData = getDataLoader(configMain.mode, configMain, useAllPts, shuffle)
        testAE(configMain, testData)


if __name__ == '__main__':
    mainAE()
