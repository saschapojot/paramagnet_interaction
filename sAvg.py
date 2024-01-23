import pickle
# from copy import deepcopy
import numpy as np
from datetime import datetime
# from multiprocessing import Pool
import matplotlib.pyplot as plt
# from scipy.sparse import kron
# from scipy.sparse import lil_matrix
# from scipy.sparse import eye
# from scipy.sparse import diags
# from scipy.linalg import eigh

#This script computes avgerage value of s over MC steps
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
        self.chemPotAll = []
        self.TEq=1000
        self.equilibrium=False


tLoadStart=datetime.now()

TAll=[0.01,0.1,1,10,100]
t=0.4
J=-2.5
g=0.05
sAvg=[]
for T in TAll:
    inFilePrefix = "T" + str(T) + "t" + str(t) + "J" + str(J) + "g" + str(g)
    inPklFile = inFilePrefix + "out.pkl"
    tLoadStart = datetime.now()
    with open(inPklFile, "rb") as fptr:
        record = pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("loading time: ", tLoadEnd - tLoadStart)
    sLast = record.sAll[-5000::30]
    # sAbs=np.abs(sLast)
    smTmp=np.mean(sLast,axis=1)
    # print(len(smTmp))
    sVal=np.mean(np.abs(smTmp))
    sAvg.append(sVal)




print(sAvg)

fig=plt.figure()
ax = fig.add_subplot(2, 1, 1)
plt.plot(TAll,sAvg)
plt.xlabel("$T$")
plt.ylabel("spin")
plt.xticks(TAll)
ax.set_xscale('log')
plt.savefig("sAvg.png")