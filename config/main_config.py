# config class
# eventually take config values using command line args
class MainConfig(object):
    def __init__(self):
        self.minNumParts = 2
        self.maxNumParts = 9
        self.dataRoot = "data"
        self.srcDir = "data/Chair"
        self.useAllPts = False
        self.resolution = 16
        self.mode = "train"

        self.ptsBatchSize = 16 * 16 * 16 * 4
        self.batchSize = 40

        self.numWorkers = 8
        self.epochs = 250
        self.cont = False

        self.nLayersE = 5  # 5
        self.efDim = 32

        self.nLayersD = 6  # 6
        self.dfDim = 128

        self.zDim = 128  # 128
        self.hiddenSize = 256

        self.boxparamSize = 6

        self.ckpt = 'latest'
        self.vis = True
        self.visFrequency = 50
        self.valFrequency = 50
        self.saveFrequency = 50
        self.saveDir = "results/partae"
        self.saveFormat = "voxel"
        self.logDir = "log/"
        self.modelDir = "model_partae"
        self.chkpDir = "chkt_dir/partae"
        self.parallel = False
        self.lr = 5e-4
        self.lrStep = 230

        self.rec = True # to reconstruct test data
        self.byPart = False # output shape is segmented into parts or not
