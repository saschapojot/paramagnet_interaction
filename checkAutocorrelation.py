import pickle

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import statsmodels.api as sm


blkSize=100
blkNum=50
class computationData:#holding computational results to be dumped using pickle
    def __init__(self):
        # self.T=TEst
        self.blkSize=blkSize
        self.blkNum=blkNum
        self.data=[]
        self.sAll=[]
        self.EAvgAll=[]
        self.TEq=1000
        self.equilibrium=False

tLoadStart=datetime.now()
beta=0.1
inPklFile="./lagbeta"+str(beta)+"out.pkl"

with open(inPklFile,"rb") as fptr:
    record=pickle.load(fptr)
tLoadEnd=datetime.now()
print("loading time: ",tLoadEnd-tLoadStart)

tryBlkMultiple=3
def checkAutc(sAll):
    """

    :param sAll: contaning s vectors as rows
    :return: checking autocorrelation
    """
    lastNum=blkNum*blkSize*5
    sAllLast=np.array(sAll[-lastNum:])
    nCol=len(sAllLast[0,:])

    acfAll=[]
    lag=32
    for i in range(0,nCol):
        acfi=sm.tsa.acf(sAllLast[0::blkSize*tryBlkMultiple,i],nlags=lag)

        acfAll.append(acfi)

    return acfAll
tAutStart=datetime.now()
acfAll=checkAutc(record.sAll)


tAutEnd=datetime.now()
print("checking autocorrelation: ",tAutEnd-tAutStart)
fig=plt.figure()
for i in range(0,len(acfAll)):
    plt.plot(acfAll[i])

plt.xlabel("lag")
plt.ylabel("autoccorrelation of $s$")
plt.savefig("autoc.png")

