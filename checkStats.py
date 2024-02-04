import numpy as np
import re

import glob
import pickle


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
def autc(x,k):
    """

    :param x: array
    :param k: lag length
    :return: autocorrelation of x with lag k
    """
    x=np.array(x)
    if len(x)==0:
        raise ValueError("x is null")
    if len(x)==1:#only 1 element
        return 1
    meanx = np.mean(x)
    diffxmean=x-meanx #all values are the same
    if np.linalg.norm(diffxmean,ord=2)<1e-13:
        return 1
    numerator = np.sum((x[k:] - meanx) * (x[:-k] - meanx))
    denominator = np.sum(diffxmean * diffxmean)

    return numerator/denominator




part=7
pklFileNames=[]
TValsAll=[]
tValsAll=[]
JValsAll=[]
gValsAll=[]
inDir="./part"+str(part)+"/"
for file in glob.glob(inDir+"*.pkl"):
    pklFileNames.append(file)
    #search T value
    matchT=re.search(r"T(-?\d+(\.\d+)?)t",file)
    if matchT:
        TValsAll.append(matchT.group(1))
    #search t values
    matcht=re.search(r"t(-?\d+(\.\d+)?)J",file)
    if matcht:
        tValsAll.append(matcht.group(1))
    #search J values
    matchJ=re.search(r"J(-?\d+(\.\d+)?)g",file)
    if matchJ:
        JValsAll.append(matchJ.group(1))
    #search g values
    matchg=re.search(r"g(-?\d+(\.\d+)?)rand",file)
    if matchg:
        gValsAll.append(matchg.group(1))


val0=(len(TValsAll)-len(tValsAll))**2\
    +(len(TValsAll)-len(tValsAll))**2\
    +(len(TValsAll)-len(JValsAll))**2\
    +(len(TValsAll)-len(gValsAll))**2\
    +(len(TValsAll)-len(pklFileNames))**2

if val0!=0:
    raise ValueError("unequal length.")


def str2float(valList):
    ret=[float(strTmp) for strTmp in valList]
    return ret
TValsAll=str2float(TValsAll)
tValsAll=str2float(tValsAll)
JValsAll=str2float(JValsAll)
gValsAll=str2float(gValsAll)
#sort temperatures
T_inds=np.argsort(TValsAll)
TValsAll=[TValsAll[ind] for ind in T_inds]
tValsAll=[tValsAll[ind] for ind in T_inds]
JValsAll=[JValsAll[ind] for ind in T_inds]
gValsAll=[gValsAll[ind] for ind in T_inds]
pklFileNames=[pklFileNames[ind] for ind in T_inds]


ldInd=-5
inPklFileName=pklFileNames[ldInd]
