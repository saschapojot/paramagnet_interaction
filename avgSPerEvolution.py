import pickle
# from copy import deepcopy
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import re

# this script computes the evolution of  average of s over 1 supercell through the MC process

class computationData:#holding computational results to be dumped using pickle
    def __init__(self):
        # self.T=TEst
        # self.blkSize=blkSize
        # self.blkNum=blkNum
        self.data=[]
        self.sAll=[]
        self.EAvgAll=[]
        self.chemPotAll = []
        self.loop=1000
        self.equilibrium=False



part=5
pklFileNames=[]
TValsAll=[]
inDir="./part"+str(part)+"/"
for file in glob.glob(inDir+"*.pkl"):
    pklFileNames.append(file)
    # search T value
    matchT = re.search(r"T(-?\d+(\.\d+)?)t", file)
    if matchT:
        TValsAll.append(matchT.group(1))



val0=(len(TValsAll)-len(pklFileNames))**2

if val0!=0:
    raise ValueError("unequal length.")

def str2float(valList):
    ret=[float(strTmp) for strTmp in valList]
    return ret


TValsAll=str2float(TValsAll)

#sort temperatures
T_inds=np.argsort(TValsAll)

pklFileNames=[pklFileNames[ind] for ind in T_inds]

figDir=inDir+"/sEvo/"
Path(figDir).mkdir(parents=True, exist_ok=True)
lastNum=20000#use the last lastNum configurations
separation=100#separation of the used configurations


for i in range(0,len(pklFileNames)):
    inPklFileName = pklFileNames[i]
    tLoadStart = datetime.now()
    with open(inPklFileName, "rb") as fptr:
        record = pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("loading time: ", tLoadEnd - tLoadStart)
    sAllMean=np.mean(record.sAll,axis=1)
    #plt

    fig,ax=plt.subplots()
    ax.plot(range(0,len(sAllMean)),sAllMean,color="black")
    ax.set_ylabel("$\\frac{1}{L}\sum_{i=0}^{L-1}s_{i}$")
    ax.set_xlabel("MC step")
    plt.savefig(figDir+"rec"+str(i)+"temp"+str(TValsAll[i]))
    plt.close()
