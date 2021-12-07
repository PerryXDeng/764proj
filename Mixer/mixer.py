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
# seed the pseudorandom number generator
import random

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


colors = [[0, 0, 255, 255],  # blue
          [0, 255, 0, 255],  # green
          [255, 0, 0, 255],  # red
          [255, 255, 0, 255],  # yellow
          [0, 255, 255, 255],  # cyan
          [255, 0, 255, 255],  # Magenta
          [160, 32, 240, 255],  # purple
          [255, 255, 240, 255]]  # ivory


def L2Distance(vec1, vec2):
    sum = 0
    for i in range(128):
        sum = sum + ((vec1[i] - vec2[i]) * (vec1[i] - vec2[i]))
    return math.sqrt(sum)


# pathJson = "data/parts_json"
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
                 size, encoding, type):
        self.voxel = voxel
        self.batchPoints = batchPoints
        self.batchValues = batchValues
        self.scale = scale
        self.translation = translation
        self.size = size
        self.encoding = encoding
        self.type = type


class Chair:
    def __init__(self, nParts, partList):
        self.nParts = nParts
        self.partList = partList


class Mixer:
    def __init__(self, resolution=64, agent=None, config=None, listIn=None, templatePath=None,
                 templateID=None, mode="project", KMM=3, restriction="strict",depthFirst=None):

        self.allChairs = []
        self.resolution = resolution
        self.agentAE = agent
        self.config = config
        self.list = listIn
        self.listLength = len(listIn)
        self.loadAllChairs()
        self.labels = None
        self.mode = mode
        self.KMM = KMM
        self.templatePath = templatePath
        self.templateID = templateID
        self.templateChair = None
        self.loadTemplate()
        self.restriction = restriction
        self.depthFirst = depthFirst

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
                        translation[i], size[i], encoding[i], 4)
            partList.append(part)
            # read type of part

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
            elif partName == "Chair Base":
                for j in range(i, nParts):
                    partList[j].type = 3

        chair = Chair(nParts, partList)
        return chair

    def loadAllChairs(self):
        for i in range(self.listLength):
            path = "data/Chair/" + str(self.list[i]) + ".h5"
            chair = self.loadOneChair(path, self.list[i], pathJson)
            if chair == False:
                continue
            self.allChairs.append(chair)

    def loadTemplate(self):
        path = os.path.join(self.templatePath, str(self.templateID) + ".h5")
        self.templateChair = self.loadOneChair(path, self.templateID, pathJson)

    def findNearestUsingL2(self, partNum, curModel, numParts):

        modID = -1
        modPart = -1
        partRef = self.allChairs[curModel].partList[partNum]
        minI = 1e6
        iList = []
        for j in range(self.listLength):
            if j == curModel:
                continue
            for i in range(numParts):
                if i >= self.allChairs[j].nParts:
                    continue
                partTest = self.allChairs[j].partList[i]
                if partTest.type != partRef.type:
                    continue
                l2dist = L2Distance(partRef.encoding, partTest.encoding)

                total = l2dist
                iList.append([j, i])
                if total < minI:
                    minI = total
                    modID = j
                    modPart = i

        return modID, modPart

    # projectEncoding(templateCurPart.encoding, templatePrePart.encoding, chairChosenEnc)
    def projectEncoding(self, A, B, P):
        # A + dot(AP,AB) / dot(AB,AB) * AB
        AB = B - A
        AP = P - A
        dotP = np.dot(AB, AP)
        den = np.dot(AB, AB)
        res = A + dotP / den * AB
        # chairChosenEnc, templateCurPart.encoding, templatePrePart.encoding

        return res

    def findListtUsingL2(self, partIdx, partType, chairID=None):

        chairIDList = []
        partIDList = []

        for i in range(self.listLength):
            for j in range(self.allChairs[i].nParts):
                partChair = self.allChairs[i].partList[j]
                # only look at the back
                if partChair.type == partType:
                    chairIDList.append(i)
                    partIDList.append(j)

        L2List = []
        partEncList = []

        if chairID == None:
            tempEnc = self.templateChair.partList[partIdx].encoding
        else:
            tempEnc = self.allChairs[chairID].partList[partIdx].encoding

        L2_Model_Part = []
        for i in range(len(chairIDList)):
            ind = chairIDList[i]
            partEnc = self.allChairs[ind].partList[partIDList[i]].encoding
            partEncList.append(partEnc)

            l2dist = L2Distance(partEnc, tempEnc)
            L2List.append(l2dist)
            L2_Model_Part.append([l2dist, ind, partIDList[i]])

        L2_Model_Part.sort(key=operator.itemgetter(0))

        return L2_Model_Part, partEncList, tempEnc

    def findBack(self, partIdx):

        # templatePartID = 0
        # find the relevant chairs and their corresponding parts

        L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(partIdx, partType=0)
        # only take top 3 and take their avg based on their distance

        depth = self.depthFirst

        if self.mode == "replace":
        # if True:
            r = randrange(0, depth)
            encBack = partEncList[r]
            backListChair = [L2_Model_Part[r][1]]
            backListPart = [L2_Model_Part[r][2]]
        else:
            r = randrange(0, depth)
            encBack = partEncList[r]
            backListChair = [L2_Model_Part[r][1]]
            backListPart = [L2_Model_Part[r][2]]

            # centre, new, old
            encBack = 0.10 * tempEnc + 0.90 * encBack

            # weightsBack = np.random.dirichlet(np.ones(4), size=1)[0]
            # rlist = [randrange(0, depth), randrange(0, depth), randrange(0, depth)]
            #
            # backListChair = [L2_Model_Part[rlist[0]][1], L2_Model_Part[rlist[1]][1], L2_Model_Part[rlist[2]][1]]
            # backListPart = [L2_Model_Part[rlist[0]][2], L2_Model_Part[rlist[1]][2], L2_Model_Part[rlist[2]][2]]
            # encBack = (weightsBack[0]) * partEncList[0] + (weightsBack[1]) * partEncList[1] \
            #           + (weightsBack[2]) * partEncList[2] + (weightsBack[3]) * tempEnc

        return encBack, L2_Model_Part, backListChair, backListPart

    def mixNmatchImproved(self, numModels=1, outDir="results/mix"):

        self.outputTemplate(outDir)

        for modNum in range(numModels):
            sequenceGeo = []
            sequenceScale = []
            sequenceTrans = []
            nparts = 0

            # make thy template here
            # template = randrange(self.listLength)
            # template = 0
            # numParts = self.allChairs[template].nParts
            numParts = self.templateChair.nParts
            chairListUse = {}
            partListUse = {}

            # find the back to use
            encBack, L2_Model_Part, backListChair, backListPart = self.findBack(0)
            partTemplate = self.templateChair.partList[0]
            translation = partTemplate.translation
            scale = partTemplate.scale

            sequenceGeo.append(encBack)
            sequenceScale.append(scale)
            sequenceTrans.append(translation)
            nparts = nparts + 1
            # add the three chairs used for back
            chairListUse[0] = backListChair
            partListUse[0] = backListPart

            resEncArm = []
            resEncLeg = []
            armDone = False
            legDone = False
            # go over all generating the other pats

            for restI in range(1, numParts):
                lastChair = self.allChairs[(chairListUse[restI - 1][0])]
                templatePrePart = self.templateChair.partList[restI - 1]
                templateCurPart = self.templateChair.partList[restI]
                partType = templateCurPart.type

                # find if last chair used had this part in it
                partIdx = -1
                for lastI in range(lastChair.nParts):
                    if lastChair.partList[lastI].type == partType:
                        partIdx = lastI
                        partIDxbackup = partIdx
                        break

                # we did not find the previous chair had same part, use template as base instead
                if self.restriction == "strict":
                    partIdx = -1



                if partIdx == -1:
                    # find three closest same type parts to my current part, then choose one randomly
                    L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(restI, partType=partType)
                    randIndPart = random.randint(0, self.KMM)
                    chairChosen = L2_Model_Part[randIndPart][1]
                    partChosen = L2_Model_Part[randIndPart][2]
                else:
                    L2_Model_Part, partEncList, tempEnc = self.findListtUsingL2(partIdx, partType=partType,
                                                                                chairID=chairListUse[restI - 1][0])
                    randIndPart = random.randint(0, self.KMM)
                    chairChosen = L2_Model_Part[randIndPart][1]
                    partChosen = L2_Model_Part[randIndPart][2]

                if partType == 1:
                    chairChosen = chairListUse[restI - 1][0]
                    partChosen = partIDxbackup

                # project the encoding of chosen part to the vector
                chairChosenPart = self.allChairs[chairChosen].partList[partChosen]
                chairChosenEnc = chairChosenPart.encoding

                if self.restriction == "strict":
                    partIdx = -1


                # A, B , P
                encMiddle = lastChair.partList[partIdx].encoding
                origin = encMiddle * 0
                if self.restriction == "strict":
                    resEnc = self.projectEncoding(origin, chairChosenEnc, templateCurPart.encoding)
                    # a = 0.8
                    # resEnc = a * chairChosenEnc + (1-a) * templateCurPart.encoding
                if self.restriction == "relax":
                    resEnc = self.projectEncoding(origin, chairChosenEnc, encMiddle)
                    # a = 0.8
                    # resEnc = a * chairChosenEnc + (1 - a) * encMiddle


                # save the used chair for further use
                chairListUse[restI] = [chairChosen]
                partListUse[restI] = [partChosen]

                if self.mode == "replace":
                    resEnc = chairChosenEnc

                translation = templateCurPart.translation
                scale = templateCurPart.scale
                # scalePart = chairChosenPart.scale
                # scale = (scale + scalePart) / 2

                if partType == 2:  # armchair
                    if armDone:
                        resEnc = resEncArm
                    else:
                        resEncArm = resEnc
                        armDone = True

                # if partType == 3:  # legchair
                #     if legDone:
                #         resEnc = resEncLeg
                #     else:
                #         resEncLeg = resEnc
                #         legDone = True

                sequenceGeo.append(resEnc)
                sequenceScale.append(scale)
                sequenceTrans.append(translation)



                nparts = nparts + 1

            # self.outputThyModel(template, outDir, str(modNum) + "_template")

            # generate the sequence
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

                # voxelIn = sdf2voxel(pts, out, voxDim=voxDim)
                vertices, triangles = libmcubes.marching_cubes(voxelIn, 0)
                inCol = colors[genI % len(colors)]
                mesh = trimesh.Trimesh(vertices, triangles, face_colors=inCol)
                mesh.apply_translation((-32, -32, -32))

                scale = sequenceScale[genI]
                mesh.apply_scale([scale, scale, scale])
                translation = sequenceTrans[genI]
                mesh.apply_translation(translation)

                shape_mesh.append(mesh)

            modelName = "template_" + str(self.templateID) + "model_" + str(modNum) + ".obj"
            shape_mesh = trimesh.util.concatenate(shape_mesh)
            savePath = os.path.join(outDir, modelName)
            shape_mesh.export(savePath, file_type='obj')

            ll = []
            for i in range(numParts):
                ll.append(chairListUse[i][0])
            print(ll)


        return True

    def outputThyModel(self, modelNum, dir, name):
        model = self.allChairs[modelNum]
        numParts = model.nParts
        shape_mesh = []
        for i in range(numParts):
            voxels = self.allChairs[modelNum].partList[i].voxel
            voxels = voxels.cpu().detach().numpy()
            voxels = voxels[0]
            vertices, triangles = libmcubes.marching_cubes(voxels, 0)
            part = self.allChairs[modelNum].partList[i].type
            inCol = colors[part % len(colors)]
            mesh = trimesh.Trimesh(vertices, triangles, face_colors=inCol)
            mesh.apply_translation((-32, -32, -32))

            scale = self.allChairs[modelNum].partList[i].scale
            mesh.apply_scale([scale, scale, scale])
            translation = self.allChairs[modelNum].partList[i].translation
            mesh.apply_translation(translation)
            shape_mesh.append(mesh)

        modelName = "template_" + str(self.templateID) + name + ".obj"
        shape_mesh = trimesh.util.concatenate(shape_mesh)
        savePath = os.path.join(dir, modelName)
        shape_mesh.export(savePath, file_type='obj')

    def outputTemplate(self, dir):
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
        savePath = os.path.join(dir, modelName)
        shape_mesh.export(savePath, file_type='obj')
        return scene

