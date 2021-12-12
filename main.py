from config import getConfig
from agents import get_agent
from random import randrange
from dataload.data_utils import getIdsMode
import os
import numpy as np
import glob
import random
from Mixer.mixer import fileDoesntExist


#############
# TO CHANGE #
dataPath = "data/TestData/set3"  # change this to new data as needec
numModelsPerTemplate = 2 #increase to get more chairs, but will probably get repeats
#############


objs = os.listdir(dataPath)
numTemplates = len(objs)

#output metrics
printnoAdjust = False
outputTemplate = False
seqName = True
nnValBack = 2
kmmVal = 2
lenList = 50
pathJson = "data/parts_json"  # this will be shared by both
# not usimg. so dont change
loadAllData = True  # overrides list length
useTemplate = True
useDataForTemplate = True
usePCA = False
pcaNum = 50
modeMain = True
applyRandomTest = True


mode1 = "replace"  # replace or project
mode2 = "strict"  # strict or relax

if useDataForTemplate:
    templatePath = dataPath
else:
    templatePath = "data/Chair"

outputDir = "results/mix"
files = glob.glob(os.path.join(outputDir, "*"))
for f in files:
    os.remove(f)



if modeMain:
    from Mixer.mixer import Mixer
else:
    from Mixer.mixer_backup import Mixer

def main():
    obbInfo = {}
    config = getConfig()
    agent = get_agent("partae", config)
    agent.loadChkPt(config.ckpt)
    config.resolution = 64
    list = []

    objs = os.listdir(dataPath)
    num = len(objs)
    itemList = random.sample(range(num), num)

    if loadAllData:
        listLength = num
    else:
        listLength = lenList

    for i in range(listLength):
        item = itemList[i]
        filename = objs[item]
        path = os.path.join(dataPath, filename)
        if fileDoesntExist(path):
            continue
        path = os.path.join(pathJson, str(filename.replace(".h5", "")), "result.json")
        if fileDoesntExist(path):
            continue
        list.append(int(filename.replace(".h5", "")))
        if len(list) == listLength:
            break

    if len(list) == 0:
        print("Error")
        return False

    for kk in range(numTemplates):

        if not useDataForTemplate:
            objs = os.listdir(templatePath)
            foundTemplate = False
            while not foundTemplate:
                template = randrange(0, len(objs))
                templateID = objs[template].replace(".h5", "")
                path = os.path.join(pathJson, str(templateID), "result.json")
                if not fileDoesntExist(path):
                    break

            if not useTemplate:
                if templateID in list:
                    list.remove(templateID)
        else:
            objs = os.listdir(dataPath)
            objs.sort()
            templateID = objs[kk].replace(".h5", "")

        print(templateID)
        mixer = Mixer(agent=agent, config=config, listIn=list, dataPath=dataPath, templatePath=templatePath,
                      templateID=templateID, mode=mode1, KMM=kmmVal, restriction=mode2, startTemp=False,
                      depthFirst=nnValBack,
                      useTemplate=useTemplate, thresholdNN=3, pcaDataTrainNum=6000, pcaComp=pcaNum,usePCA=usePCA,
                      applyRandomTest=applyRandomTest,noAdjust=printnoAdjust,seqName=seqName)

        # mixer.findPCAofTrain()
        mixer.loadPCAInfo()
        if outputTemplate:
            mixer.outputTemplate(outDir=outputDir)
        mixer.mixNmatchImproved(numModels=numModelsPerTemplate, outDir=outputDir, list=list)

    # replace strict knn=15 depthFirst=10
    # replace relax knn = 3 depthFirst=5
    # project strict knn = 5 depth = 5
    # project relax knn = 5 depth = 5


if __name__ == '__main__':
    main()
