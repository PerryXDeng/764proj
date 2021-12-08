from config import getConfig
from Mixer.mixer import Mixer
from agents import get_agent
from random import randrange
from dataload.data_utils import getIdsMode
import os
import numpy as np
import glob
import random

def fileDoesntExist(path):
    if os.path.exists(path):
        return False
    else:
        return True


dataPath = "data/TestData"  # change this to new data as needec
pathJson = "data/parts_json"  # this will be shared by both
numTemplates = 5
numModelsPerTemplate = 2
listLength = 20  # per template
nnValBack = 4  # keep 2-3 values less than list length
kmmVal = 4  # keep 2-3 values  less than list lenght
useTemplate = False
useDataForTemplate = False
loadAllData = True #overrides list length

mode1 = "replace" #replace or project
mode2 = "relax" #strict or relax

if useDataForTemplate:
    templatePath = "data/TestData"
else:
    templatePath = "data/Chair"


outputDir = "results/mix"
files = glob.glob(os.path.join(outputDir, "*"))
for f in files:
    os.remove(f)


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
        listLength = num-1

    for i in range(listLength):
        item = itemList[i]
        filename = objs[item]
        path = os.path.join(dataPath, filename)
        if fileDoesntExist(path):
            continue
        path = os.path.join(pathJson, str(filename.replace(".h5","")), "result.json")
        if fileDoesntExist(path):
            continue
        list.append(int(filename.replace(".h5","")))
        if len(list) == listLength:
            break

    if len(list) == 0:
        print("Error")
        return False

    for kk in range(numTemplates):

        objs = os.listdir(templatePath)
        foundTemplate = False
        while not foundTemplate:
            template = randrange(0, len(objs))
            templateID = objs[template].replace(".h5","")
            path = os.path.join(pathJson, str(templateID), "result.json")
            if not fileDoesntExist(path):
                break

        if not useTemplate:
            if templateID in list:
                list.remove(templateID)

        print(templateID)
        mixer = Mixer(agent=agent, config=config, listIn=list, templatePath=templatePath,
                      templateID=templateID, mode=mode1, KMM=kmmVal, restriction=mode2, depthFirst=nnValBack,
                      useTemplate=useTemplate, thresholdNN=2)
        mixer.outputTemplate(dir=outputDir)
        mixer.mixNmatchImproved(numModels=numModelsPerTemplate, outDir=outputDir, list=list)

    # replace strict knn=15 depthFirst=10
    # replace relax knn = 3 depthFirst=5
    # project strict knn = 5 depth = 5
    # project relax knn = 5 depth = 5


if __name__ == '__main__':
    main()
