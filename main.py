from config import getConfig
from dataload import getDataLoader
from networks import encoderTest, encoderTrain


def main():
    # load config parameters here
    config = getConfig()

    # if training selected
    # here the autoencoder will be trained
    if config.mode == "train":
        shuffle = True
        trainData = getDataLoader(config.mode, config, shuffle)
        encoderTrain(trainData)

    # if testing selected
    # here the encoder is used to generate latent code only, the decoder isnt used
    if config.mode == "test":
        shuffle = False
        testData = getDataLoader(config.mode, config, shuffle)
        encoderTest(testData)


if __name__ == '__main__':
    main()
