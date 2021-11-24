# config class
# eventually take config values using command line args
class MainConfig(object):
    def __init__(self):
        self.numParts = 6
        self.srcDir = "data/Chair"
        self.ptsBatchSize = 24
        self.useAllPts = False
        self.resolution = 64
        self.mode = "train"
        self.batchSize = 40
        self.numWorkers = 1
        self.epochs = 250
        self.cont = False
        self.ckpt = 0

        self.nLayersE = 6
        self.efDim = 64
        self.nLayersD = 6
        self.dfDim = 64
        self.zDim = 64
        self.vis = False
        self.visFrequency = 10
        self.valFrequency = 10
        self.saveDir = "result/partae"
        self.saveFormat = "mesh"
