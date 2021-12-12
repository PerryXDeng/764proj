import os
from dataload.data_utils import loadH5Full, getIdsMode
from random import randrange
import trimesh
import mcubes as libmcubes
import math
import torch
import numpy as np
from scipy.io import loadmat
from utils import sdf2voxel, projVoxelXYZ, partsdf2voxel
import json
import operator
import random
import time
from sklearn.decomposition import PCA
import pandas as pd
import pickle

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import matplotlib.pyplot as plt
from PIL import Image
import io
from pyglet import gl
import scipy.misc
import imageio
import pyvista as pv
from pyvista import examples

# Type of parts
# 0 - back
# 1 - seat
# 2 - armchair
# 3 - leg
# 4 - miscelannous

# types of legs
# tells number of legs and caps at 4
from itertools import groupby


def fileDoesntExist(path):
    if os.path.exists(path):
        return False
    else:
        return True


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)



colors = [[0, 0, 255, 255],
          [0, 255, 0, 255],
          [255, 0, 0, 255],
          [255, 255, 0, 255]]

grey = [128, 128, 128, 255]


def L2Distance(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        sum = sum + ((vec1[i] - vec2[i]) * (vec1[i] - vec2[i]))
    return math.sqrt(sum)


pathJson = "data/parts_json"


def readJsonParts(path, chairID):
    listNames = []
    path = os.path.join(path, str(chairID), "result.json")
    f = open(path)
    data = json.load(f)
    for i in data[0]['children']:
        listNames.append(i['text'])
    f.close()
    return listNames


class Part:
    def __init__(self, voxel, batchPoints, batchValues, scale, translation,
                 size, encoding, type, legNum):
        self.voxel = voxel
        self.batchPoints = batchPoints
        self.batchValues = batchValues
        self.scale = scale
        self.translation = translation
        self.size = size
        self.encoding = encoding
        self.type = type
        self.legNum = legNum


class Chair:
    def __init__(self, ID, nParts, partList, numLegs, firstArm, firstLeg):
        self.ID = ID
        self.nParts = nParts
        self.partList = partList
        self.numLegs = numLegs
        self.firstArm = firstArm
        self.firstLeg = firstLeg


class Mixer:
    def __init__(self, resolution=64, agent=None, config=None, listIn=None, dataPath=None, templatePath=None,
                 templateID=None, mode="project", KMM=3, restriction="strict", startTemp=None, depthFirst=None,
                 useTemplate=False, thresholdNN=-1, pcaDataTrainNum=None, pcaComp=2, usePCA=False,
                 applyRandomTest=True, noAdjust=False, seqName=None):

        self.dataPath = dataPath
        self.allChairs = []
        self.allTrainChairs = []
        self.allDataChairs = []
        self.resolution = resolution
        self.agentAE = agent
        self.config = config
        self.list = listIn
        self.listLength = len(listIn)

        self.labels = None
        self.mode = mode
        self.KMM = KMM
        self.templatePath = templatePath
        self.templateID = templateID
        self.templateChair = None
        self.restriction = restriction
        self.depthFirst = depthFirst
        self.useTemplate = useTemplate
        self.thresholdNN = thresholdNN
        self.legModel = -1
        self.pcaInfo = None
        self.pcaDataTrainNum = pcaDataTrainNum
        self.pcaComp = pcaComp
        self.startTemp = startTemp
        self.tempLoc = -1
        self.usePCA = usePCA
        self.applyRandomTest = applyRandomTest
        self.chairListRT = []
        self.loadTemplate()
        self.loadAllChairs()
        self.printNoAdjust = noAdjust
        self.seqName = seqName

    def loadPCAInfo(self):
        path = os.path.join("chkt_dir/pca", "pca_" + str(self.pcaComp) + ".pkl")
        f = open(path, 'rb')
        self.pcaInfo = pickle.load(f)
        f.close()

    def findPCAofTrain(self):
        # load encodings of all all chairs
        self.loadAllTrainChairs()
        encodings = {}
        pcaList = []
        for i in range(len(self.allTrainChairs)):
            chair = self.allTrainChairs[i]
            for j in range(chair.nParts):
                part = chair.partList[j]
                if encodings.get(part.type) is None:
                    encodings[part.type] = []
                encodings[part.type].append(part.encoding)

        ncomponents = [2, 5, 10, 15, 20, 50]
        for j in range(len(ncomponents)):
            pcaList = []
            for i in range(4):
                x = encodings[i]
                pca = PCA(n_components=ncomponents[j])
                pca.fit_transform(x)
                pcaList.append(pca)

            # Saving the objects:
            filename = "pca_" + str(ncomponents[j]) + ".pkl"
            path = os.path.join("chkt_dir/pca", filename)
            with open(path, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(pcaList, f)

    def loadOneChairEncodeOnly(self, path, chairID, jsonPath):
        # nPartsA = data['n_parts'].cpu().detach().numpy().flatten()
        # nParts = nPartsA[0]
        # read json file of chair
        listParts = readJsonParts(jsonPath, chairID)

        nParts, partVoxel, dataPoints, dataVals, scale, translation, size = loadH5Full(path, resolution=self.resolution)
        batchVoxels = torch.tensor(partVoxel.astype(np.float), dtype=torch.float32).unsqueeze(1)  # (1, dim, dim, dim)
        partList = []
        inVox3d = batchVoxels
        encoding = self.agentAE.net.encoder(inVox3d.cuda()).cpu().detach().numpy()
        for i in range(nParts):
            part = Part(inVox3d[i], None, None, None,
                        None, None, encoding[i], 4, 0)
            partList.append(part)

        nLegs = 0
        firstArm = -1
        firstLeg = -1
        for i in range(len(listParts)):
            partName = listParts[i]
            if i >= len(partList):
                continue
            if partName == "Chair Back":
                partList[i].type = 0
            elif partName == "Chair Seat":
                partList[i].type = 1
            elif partName == "Chair Arm":
                partList[i].type = 2
                firstArm = i if firstArm == -1 else firstArm
            elif partName == "Chair Base":
                for j in range(i, nParts):
                    partList[j].type = 3
                    nLegs = nLegs + 1
                    partList[j].legNum = nLegs
                firstLeg = i

        numLegs = nLegs if nLegs < 4 else 4
        chair = Chair(chairID, nParts, partList, numLegs, firstArm, firstLeg)
        return chair

    def loadOneChair(self, path, chairID, jsonPath):
        # nPartsA = data['n_parts'].cpu().detach().numpy().flatten()
        # nParts = nPartsA[0]
        # read json file of chair
        listParts = readJsonParts(jsonPath, chairID)

        nParts, partVoxel, dataPoints, dataVals, scale, translation, size = loadH5Full(path, resolution=self.resolution)
        batchVoxels = torch.tensor(partVoxel.astype(np.float), dtype=torch.float32).unsqueeze(1)  # (1, dim, dim, dim)
        batchPoints = torch.tensor(dataPoints, dtype=torch.float32)  # (points_batch_size, 3)
        batchValues = torch.tensor(dataVals, dtype=torch.float32)  # (points_batch_size, 1)

        partList = []
        inVox3d = batchVoxels
        encoding = self.agentAE.net.encoder(inVox3d.cuda()).cpu().detach().numpy()
        translation = translation
        scale = scale
        size = size
        for i in range(nParts):
            part = Part(inVox3d[i], batchPoints[i], batchValues[i], scale[i],
                        translation[i], size[i], encoding[i], 4, 0)
            partList.append(part)
            # read type of part

        nLegs = 0
        firstArm = -1
        firstLeg = -1
        for i in range(len(listParts)):
            partName = listParts[i]
            if i >= len(partList):
                continue
            if partName == "Chair Back":
                partList[i].type = 0
            elif partName == "Chair Seat":
                partList[i].type = 1
            elif partName == "Chair Arm":
                partList[i].type = 2
                firstArm = i if firstArm == -1 else firstArm
            elif partName == "Chair Base":
                for j in range(i, nParts):
                    partList[j].type = 3
                    nLegs = nLegs + 1
                    partList[j].legNum = nLegs
                firstLeg = i

        numLegs = nLegs if nLegs < 4 else 4

        chair = Chair(chairID, nParts, partList, numLegs, firstArm, firstLeg)
        return chair

    def loadAllChairs(self):
        for i in range(self.listLength):
            path = self.dataPath + "/" + str(self.list[i]) + ".h5"
            chair = self.loadOneChair(path, self.list[i], pathJson)
            if chair.ID == int(self.templateID):
                self.tempLoc = i
            if not chair:
                continue
            self.allChairs.append(chair)

    def loadAllTrainChairs(self):
        pathMain = "data/Chair"
        objs = os.listdir(pathMain)
        num = len(objs)
        list = random.sample(range(num), self.pcaDataTrainNum)

        for i in range(self.pcaDataTrainNum):
            item = list[i]
            filename = objs[item]
            path = os.path.join(pathJson, str(filename.replace(".h5", "")), "result.json")
            if fileDoesntExist(path):
                continue
            path = os.path.join(pathMain, filename)
            chair = self.loadOneChairEncodeOnly(path, filename.replace(".h5", ""), pathJson)
            if not chair:
                continue
            self.allTrainChairs.append(chair)

    def loadTemplate(self):
        path = os.path.join(self.templatePath, str(self.templateID) + ".h5")
        self.templateChair = self.loadOneChair(path, self.templateID, pathJson)

    # projectEncoding(templateCurPart.encoding, templatePrePart.encoding, chairChosenEnc)

    def findListtUsingL2(self, partIdx, partType, chairID=None, sendTemplate=False, randOff=False):

        templatenum = -1
        chairIDList = []
        partIDList = []
        if chairID is None:
            partTemplate = self.templateChair.partList[partIdx]
            tempEnc = partTemplate.encoding
        elif (chairID is not None) & (partIdx>=self.allChairs[chairID].nParts):
            partTemplate = self.templateChair.partList[partIdx]
            tempEnc = partTemplate.encoding
        else:
            tempEnc = self.allChairs[chairID].partList[partIdx].encoding
            partTemplate = self.allChairs[chairID].partList[partIdx]
        penalty = []
        for i in range(self.listLength):
            if self.allChairs[i].ID == int(self.templateID):
                templatenum = i
                if (not self.useTemplate) | (not self.applyRandomTest):
                    continue
            for j in range(self.allChairs[i].nParts):
                partChair = self.allChairs[i].partList[j]
                # only look at the back
                if partChair.type == partType:
                    chairIDList.append(i)
                    partIDList.append(j)
                    penalty.append(0)
                    if partType == 3:
                        legDiff = 0
                        diffLoc = 0
                        diffLegs = 0
                        curChair = self.allChairs[i]
                        if (self.legModel != -1) & (i != self.legModel):
                            legDiff = 1
                            if abs(partTemplate.legNum - partChair.legNum) != 0:
                                diffLoc = 1
                        if abs(self.templateChair.numLegs - curChair.numLegs) !=0:
                            diffLegs = 1
                        penalty[len(penalty) - 1] = 0.5 * legDiff + 0.5 * diffLoc

        L2List = []
        partEncList = []

        L2_Model_Part = []
        for i in range(len(chairIDList)):
            ind = chairIDList[i]
            partEnc = self.allChairs[ind].partList[partIDList[i]].encoding
            partEncList.append(partEnc)
            l2dist = L2Distance(partEnc, tempEnc)
            l2dist = l2dist + penalty[i]
            if self.usePCA:
                pca = self.pcaInfo[partType]
                n1 = pca.transform(partEnc.reshape(1, -1))
                n2 = pca.transform(tempEnc.reshape(1, -1))
                pcaDist = L2Distance(n1[0], n2[0])
                l2dist = pcaDist + penalty[i]

            L2List.append(l2dist)
            L2_Model_Part.append([l2dist, ind, partIDList[i]])

        L2_Model_Part.sort(key=operator.itemgetter(0))

        hit = []
        if self.applyRandomTest & (not randOff):
            for outer in range(len(L2_Model_Part)):
                hits = 0
                for inner in range(len(self.chairListRT)):
                    if L2_Model_Part[outer][1] == self.chairListRT[inner]:
                        hits = hits + 1
                hit.append(hits)

            for outer in range(len(L2_Model_Part)):
                L2_Model_Part[outer][0] = L2_Model_Part[outer][0] + 0.5 * hit[outer]

        L2_Model_Part.sort(key=operator.itemgetter(0))

        if sendTemplate & self.useTemplate:
            L2_Model_Part.clear()
            L2_Model_Part.append([1, templatenum, partIdx])
        elif len(L2_Model_Part) == 0 & self.useTemplate:
            L2_Model_Part.append([1, templatenum, partIdx])
        elif (L2_Model_Part[0][0] > self.thresholdNN) & self.useTemplate & (not self.applyRandomTest):
            L2_Model_Part.clear()
            L2_Model_Part.append([1, templatenum, partIdx])

        return L2_Model_Part, partEncList, tempEnc

    def findBack(self, partIdx):
        L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(partIdx, partType=0, sendTemplate=self.startTemp)

        depth = self.depthFirst

        if self.mode == "replace":
            r = randrange(0, depth)
            r = r if r < len(L2_Model_Part) else 0
            encBack = partEncList[r]
            backListChair = [L2_Model_Part[r][1]]
            backListPart = [L2_Model_Part[r][2]]
        else:
            r = randrange(0, depth)
            r = r if r < len(L2_Model_Part) else 0
            encBack = partEncList[r]
            backListChair = [L2_Model_Part[r][1]]
            backListPart = [L2_Model_Part[r][2]]
            encBack = 0.25 * tempEnc + 0.85 * encBack

        return encBack, L2_Model_Part, backListChair, backListPart

    def mixNmatchImproved(self, numModels=1, outDir="results/mix", list=None):


        global chairChosen, partChosen
        for modNum in range(numModels):
            rCol = 236
            gCol = 196
            bCol = 188
            sequenceGeo = []
            sequenceScale = []
            sequenceTrans = []
            sequenceSize = []
            sequenceScaleSecond = []
            nparts = 0

            numParts = self.templateChair.nParts
            chairListUse = {}
            partListUse = {}

            # find a random chair for legs
            # find the legs to use
            for rangeLeg in range(self.templateChair.numLegs):
                L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(self.templateChair.firstLeg + rangeLeg, 3,
                                                                            randOff=True)
                kmm = 3 if 3 < len(L2_Model_Part) + 1 else len(L2_Model_Part)
                randIndPart = random.randint(0, kmm - 1)
                chairChosen = L2_Model_Part[randIndPart][1]
                partChosen = L2_Model_Part[randIndPart][2]

                if self.legModel == -1:
                    self.legModel = chairChosen

                chairListUse[rangeLeg] = [chairChosen]  # [self.tempLoc]
                partListUse[rangeLeg] = [partChosen]  # [chairI.firstLeg + rangeLeg]

                partTemplate = self.templateChair.partList[self.templateChair.firstLeg + rangeLeg]
                encBack = partTemplate.encoding
                translation = partTemplate.translation
                # scale = [partTemplate.scale, partTemplate.scale, partTemplate.scale]

                sequenceGeo.append(encBack)
                sequenceScale.append(partTemplate.scale[0])
                sequenceScaleSecond.append([1, 1, 1])
                sequenceTrans.append(translation)
                size = partTemplate.size
                sequenceSize.append(size)
                nparts = nparts + 1

                self.chairListRT.append(chairChosen)

            resEncArm = []
            armDone = False
            chairArm = -1
            partArm = -1
            lastChair = self.allChairs[(chairListUse[nparts - 1][0])]

            for restI in range(numParts):

                templateCurPart = self.templateChair.partList[restI]
                partType = templateCurPart.type
                if partType == 3:
                    break
                # find if last chair used had this part in it
                partIdx = -1
                for lastI in range(lastChair.nParts):
                    if lastChair.partList[lastI].type == partType:
                        partIdx = lastI
                        break

                if self.restriction == "strict":
                    partIdx = -1

                time.sleep(0.003)
                if partIdx == -1:
                    # find closest same type parts to my current part, then choose one randomly
                    L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(restI, partType=partType,randOff=False)
                    kmm = self.KMM if self.KMM < len(L2_Model_Part) + 1 else len(L2_Model_Part)
                    randIndPart = random.randint(0, kmm - 1)
                    chairChosen = L2_Model_Part[randIndPart][1]
                    partChosen = L2_Model_Part[randIndPart][2]
                else:
                    if restI == 0:
                        chairid = nparts - 1
                    else:
                        chairid = chairListUse[restI - 1][0]
                    L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(partIdx, partType=partType,
                                                                                chairID = chairid,randOff=False)
                    kmm = self.KMM if self.KMM < len(L2_Model_Part) + 1 else len(L2_Model_Part)
                    if kmm == 0:
                        randIndPart = 0
                    else:
                        randIndPart = random.randint(0, kmm - 1)
                    chairChosen = L2_Model_Part[randIndPart][1]
                    partChosen = L2_Model_Part[randIndPart][2]

                # project the encoding of chosen part to the vector
                chairChosenPart = self.allChairs[chairChosen].partList[partChosen]
                chairChosenEnc = chairChosenPart.encoding

                if self.restriction == "strict":
                    partIdx = -1

                # A, B , P
                encMiddle = lastChair.partList[partIdx].encoding
                origin = encMiddle * 0
                if self.restriction == "strict":
                    a = 0.85
                    resEnc = a * chairChosenEnc + (1 - a) * templateCurPart.encoding
                if self.restriction == "relax":
                    a = 0.85
                    resEnc = a * chairChosenEnc + (1 - a) * encMiddle

                if self.mode == "replace":
                    resEnc = chairChosenEnc

                translation = templateCurPart.translation
                scale = templateCurPart.scale

                # save the used chair for further use
                chairListUse[nparts] = [chairChosen]
                partListUse[nparts] = [partChosen]
                if partType == 2:  # armchair
                    if armDone:
                        resEnc = resEncArm
                        chairListUse[nparts] = [chairArm]
                        partListUse[nparts] = [partArm + 1]
                    else:
                        resEncArm = resEnc
                        armDone = True
                        chairArm = chairChosen
                        partArm = self.allChairs[chairChosen].firstArm
                self.chairListRT.append(chairListUse[nparts][0])

                reg = 100
                sizeTemp = templateCurPart.size
                sizeChosen = self.allChairs[chairChosen].partList[partChosen].size
                sizeRatio = (sizeTemp + reg) / (sizeChosen + reg)
                sequenceGeo.append(resEnc)
                sequenceScale.append(scale[0])

                sequenceTrans.append(translation)
                scale = sizeRatio
                sequenceScaleSecond.append(scale)
                sequenceSize.append(sizeChosen)

                lastChair = self.allChairs[(chairListUse[nparts][0])]
                nparts = nparts + 1

            ll = []
            for i in range(len(self.chairListRT)):
                ll.append(list[self.chairListRT[i]])
            self.chairListRT.clear()

            # adjScales = []
            # for adjI in range(self.templateChair.numLegs):
            #     tempPart = self.templateChair.partList[self.templateChair.firstLeg+adjI]
            #     myPart = self.allChairs[chairListUse[adjI][0]].partList[partListUse[adjI][0]]
            #     sizeTemp = tempPart.size
            #     sizePart = myPart.size
            #     scaleAdj = ((sizeTemp-sizePart)/2*64)
            #     scale = sequenceScale[adjI]
            #     adjScales.append(scale-scaleAdj)
            # for adjI in range(self.templateChair.numLegs,numParts):
            #     scaleAdj = [0,0,0]
            #     scale = sequenceScale[adjI]
            #     adjScales.append(scale-scaleAdj)

            shape_mesh = []
            voxDim = 64
            for genI in range(numParts):
                zLatent = torch.from_numpy(sequenceGeo[genI])
                zLatent = torch.unsqueeze(zLatent, 0)
                idChairs = chairListUse[genI]
                idParts = partListUse[genI]

                # aggregate points from the list
                if self.mode == "replace":
                    pointsBatch = self.allChairs[idChairs[0]].partList[idParts[0]].batchPoints.cpu().detach().numpy()
                else:
                    pointsBatch = self.templateChair.partList[genI].batchPoints.cpu().detach().numpy()
                    sizeOne = np.shape(pointsBatch)[0]
                    for ptsI in range(len(idChairs)):
                        pts = self.allChairs[idChairs[ptsI]].partList[idParts[ptsI]].batchPoints.cpu().detach().numpy()
                        pointsBatch = np.concatenate((pointsBatch, pts), axis=0)

                    if len(idChairs) > 1:
                        sizeOne = sizeOne * 2

                    index = np.random.choice(pointsBatch.shape[0], sizeOne, replace=False)
                    pointsBatch = pointsBatch[index]

                ptsBackup = pointsBatch

                pointsBatch = torch.from_numpy(pointsBatch)
                pointsBatch = torch.unsqueeze(pointsBatch, 0)

                shape_batch_size = pointsBatch.size()[0]
                point_batch_size = pointsBatch.size()[1]

                batch_z = zLatent.unsqueeze(1).repeat((1, point_batch_size, 1)).view(-1, zLatent.size(1))
                batch_points = pointsBatch.view(-1, 3)
                out = self.agentAE.net.decoder(batch_points.cuda(), batch_z.cuda())
                out = out.view((shape_batch_size, point_batch_size, -1)).cpu().detach().numpy()
                pts = ptsBackup * voxDim
                if self.mode == "replace":
                    voxelIn = self.allChairs[idChairs[0]].partList[idParts[0]].voxel
                    voxelIn = voxelIn.cpu().detach().numpy()
                    voxelIn = voxelIn[0]
                else:
                    voxelIn = sdf2voxel(pts, out, voxDim=voxDim)
                vertices, triangles = libmcubes.marching_cubes(voxelIn, 0)
                part = self.allChairs[idChairs[0]].partList[idParts[0]].type
                if all_equal(ll):
                    inCol = grey
                else:
                    inCol = colors[part % len(colors)]
                    # inCol = [rCol,gCol,bCol,255]

                mesh = trimesh.Trimesh(vertices, triangles, face_colors=inCol)
                mesh.apply_translation((-32, -32, -32))
                scale = sequenceScale[genI]
                mesh.apply_scale(scale)
                # scale = adjScales[genI]
                # mesh.apply_scale([scale[0], scale[1], scale[2]])
                translation = sequenceTrans[genI]
                mesh.apply_translation(translation)
                shape_mesh.append(mesh)
            if self.seqName:
                numR = randrange(10000, 100000)
                modelName = "model_" + str(numR) + ".obj"
            else:
                modelName = "template_" + str(self.templateID) + "model_" + str(modNum) + ".obj"
            shape_mesh = trimesh.util.concatenate(shape_mesh)
            savePath = os.path.join(outDir, modelName)
            shape_mesh.export(savePath, file_type='obj')
            print(ll)
        return ll

    def outputTemplate(self, outDir):
        model = self.templateChair
        numParts = model.nParts
        shape_mesh = []
        scene = trimesh.Scene()
        for i in range(numParts):
            voxels = self.templateChair.partList[i].voxel
            voxels = voxels.cpu().detach().numpy()
            voxels = voxels[0]
            vertices, triangles = libmcubes.marching_cubes(voxels, 0)
            part = self.templateChair.partList[i].type
            inCol = colors[part % len(colors)]
            # inCol = [229, 204, 214, 255]
            mesh = trimesh.Trimesh(vertices, triangles, face_colors=inCol)
            mesh.apply_translation((-32, -32, -32))

            scale = self.templateChair.partList[i].scale
            mesh.apply_scale([scale, scale, scale])
            translation = self.templateChair.partList[i].translation
            mesh.apply_translation(translation)
            shape_mesh.append(mesh)
            scene.add_geometry(mesh)

        modelName = "template_" + str(self.templateID) + ".obj"
        shape_mesh = trimesh.util.concatenate(shape_mesh)
        savePath = os.path.join(outDir, modelName)
        shape_mesh.export(savePath, file_type='obj')
        return scene
