from config import getConfig
from Mixer.mixer import Mixer
from agents import get_agent
from random import randrange
from dataload.data_utils import getIdsMode
import os
import numpy as np

def fileDoesntExist(path):
    if os.path.exists(path):
        return False
    else:
        return True

dataPath = "data/Chair" #change this to new data as needec
templatePath = "data/Chair"
pathJson = "data/parts_json" #this will be shared by both
numTemplates = 10

def main():
    obbInfo = {}
    config = getConfig()
    agent = get_agent("partae", config)
    agent.loadChkPt(config.ckpt)
    config.resolution = 64
    list = []
    IDS = getIdsMode("train")
    import random
    num = 4000
    listLength = 1   #per template
    itemList = random.sample(range(num), num)
    for i in range(listLength):
        item = itemList[i]
        item = IDS[item]
        filename = str(item) + ".h5"
        path = os.path.join(dataPath, filename)
        if fileDoesntExist(path):
            continue
        path = os.path.join(pathJson, str(item), "result.json")
        if fileDoesntExist(path):
            continue
        list.append(int(item))
        if len(list) == listLength:
            break

    if len(list) == 0:
        print("Error")
        return False

    for kk in range(numTemplates):
        template = randrange(0, len(list))

        templateID = list[template]
        # templateID = 2
        print(templateID)
        if templateID in list:
            list.remove(templateID)


        mixer = Mixer(agent=agent, config=config, listIn=list, templatePath=templatePath,
                      templateID = templateID, mode="replace", KMM=1, restriction="strict",depthFirst=5)
        mixer.mixNmatchImproved(numModels=5, outDir="results/mix")

    #replace strict knn=15 depthFirst=10
    #replace relax knn = 3 depthFirst=5
    #project strict knn = 5 depth = 5
    #project relax knn = 5 depth = 5

if __name__ == '__main__':
    main()
