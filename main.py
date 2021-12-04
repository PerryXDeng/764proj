from config import getConfig
from agents import get_agent
from os import listdir
from os.path import isfile, join
import os
import h5py
from dataload.data_utils import loadH5Partwise
from random import randrange
import trimesh
import mcubes as libmcubes
import math 


def L2Distance(vec1, vec2):
    sum = 0
    for i in range(vec1.size()):
        sum = sum + (vec1[i]-vec2[i])*(vec1[i]-vec2[i])
    return math.sqrt(sum)


class Part():
    def __init__(self, dataPoints16, dataVals16, dataPoints32, dataVals32, dataPoints64, dataVals64, scale, translation, size, encoding):
        self.dataPoints16 = dataPoints16
        self.dataVals16 = dataVals16
        self.dataPoints32 = dataPoints32
        self.dataVals32 = dataVals32
        self.dataPoints64 = dataPoints64
        self.dataVals64 = dataVals64
        self.scale = scale
        self.translation = translation
        self.size = size
        self.encoding = encoding

class Chair():
    def __init__(self, nParts, partList):
        self.nParts = nParts
        self.partList = partList


class Mixer():
    def __init__(self, resolution = 16, agent=None,listIDs=[]):
        self.resolution = resolution
        self.agentAE = agent
        self.listLength = self.loadAllChairs(listIDs)

    def loadOneChair(self, path):
        chair = []
        with h5py.File(path, "r") as data_file:
            nParts = data_file.attrs['n_parts']
            partList = []
            for i in range(nParts):
                nParts, partVoxel, dataPoints16, dataVals16, scale, translation, size = loadH5Partwise(path, i, resolution=16, rescale=True)
                nParts, partVoxel, dataPoints32, dataVals32, scale, translation, size = loadH5Partwise(path, i, resolution=32, rescale=True)
                nParts, partVoxel, dataPoints64, dataVals64, scale, translation, size = loadH5Partwise(path, i, resolution=64, rescale=True)
                encoding = self.agent.net.encoder(partVoxel)
                part = Part(dataPoints16, dataVals16, dataPoints32, dataVals32, dataPoints64, dataVals64, scale, translation, size, encoding)
                partList.append(part)
            chair = Chair(nParts, partList)
        return chair

    def loadAllChairs(self, listIDs):
        self.allChairs = []
        for i in range(listIDs.size()):
            path = os.path.join("data/Chair", listIDs[i] + ".h5")
            chair = self.loadOneChair(path)
            self.allChairs.append(chair)
        return listIDs.size()



    def findNearestUsingL2(self, partNum, curModel, criteria, numPartsReq):

        alpha = 0.5
        beta = 0.5
        validCandidates = []
        partRef = self.allChairs[curModel].partList[partNum]
        for i in range(self.listLength):
            if(i == curModel):
                continue;
            partTest = self.allChairs[i].partList[partNum]
            l2dist = L2Distance(partRef.encoding, partTest.encoding)

            numPartsOrg = self.allChairs[i].nParts
            tot = math.sqrt((numPartsOrg-numPartsReq)*(numPartsOrg-numPartsReq))

            total = alpha * (tot) + beta * (l2dist)

            if(total < criteria):
                validCandidates.append(i)

        if(validCandidates.size() == 0):
            return False



        k = randrange(validCandidates.size())
        return validCandidates[k]



    def mixNmatch(self, numModels = 1, L2Dist = 0.5, outDir = "results/mix"):
        # this sequence saves the latent code + bounding box info for each part 

        modelNum = 1
        modelList = []
        sequence = []
        sequenceGeo = []
        sequenceScale = []
        sequenceTrans = []
        lastModel = -1

        # start with a random part 0 

        lastModel = randrange(self.listLength)
        numParts = self.allChairs[lastModel].nParts
        part0 = self.allChairs[lastModel].partList[0]
        part0Enc = part0.encoding
        translation = part0.translation
        scale = part0.scale
        totalCode = [part0Enc, translation, scale]
        sequence.append(totalCode)
        sequenceGeo.append(part0Enc)
        sequenceScale.append(translation)
        sequenceTrans.append(scale)
        modelList.append(lastModel)

        # find encoded outputs of next part 
        # the regular next code 
        for i in range(1, numParts):
            # find next using L2 
            k = self.findNearestUsingL2(i, lastModel, 0.05, numParts)
            partIChosen = self.allChairs[k].partList[i]
            partICurrent = self.allChairs[lastModel].partList[i]

            partIExp = partICurrent
            translation = partIExp.translation
            scale = partIExp.scale
            
            partIExp = partICurrent # partIChosen
            partIEnc = partIExp.encoding
            totalCode = [partIEnc, translation, scale]

            sequence.append(totalCode)
            sequenceGeo.append(part0Enc)
            sequenceScale.append(translation)
            sequenceTrans.append(scale)
            modelList.append(lastModel)

            lastModel = k

        # generate the sequence as one chair to output directory 
        shape_mesh = []
        
        for i in range(numParts):
            
            modelUse = modelList[i]
            voxels = self.allChairs[modelUse][i]
            vertices, triangles = libmcubes.marching_cubes(voxels, 0)

            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.apply_translation((-32, -32, -32))

            scale = sequenceScale[i]
            mesh.apply_scale([scale, scale, scale])
            translation = sequenceTrans[i]
            mesh.apply_translation(translation)
            
            shape_mesh.append(mesh)

        shape_mesh = trimesh.util.concatenate(shape_mesh)
        savePath = os.path.join(outDir, "model_{}.obj".format(modelNum))
        shape_mesh.export(savePath, file_type='obj')

        return False



def main(): 
    config = getConfig()
    agent = get_agent("partae")
    agent.loadChkPt(config.ckpt)

    list = [172]
    mixer = Mixer(listIDs=list)
    #output required number of models with the given list
    mixer.mixNmatch(numModels = 1, L2Dist = 0.5 ,outDir = "results/mix")


if __name__ == '__main__':
    main()
