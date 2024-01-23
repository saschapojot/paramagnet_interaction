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


TemperaturesAll=[0.1+10*n for n in range(0,51)]

t=0.4
J=-2.5
g=0.05
tPltStart=datetime.now()
sAvgAll=[]
for i in range(0,len(TemperaturesAll)):
    T=TemperaturesAll[i]
    inFileName="T"+str(T)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"out.pkl"
    tLoadStart = datetime.now()
    with open(inFileName,"rb") as fptr:
        record=pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("finished loading "+str(i))
    print("loading time: ", tLoadEnd - tLoadStart)
    sLast=record.sAll[-5000::30]
    supcellMean=np.abs(np.mean(sLast,axis=1))
    sAvgAll.append(np.mean(supcellMean))


fig=plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.plot(TemperaturesAll,sAvgAll)

ax.set_xlabel("$T$")
ax.set_ylabel("$<s>$")
ax.set_xscale('log')
plt.savefig("sAvglogx.png")
plt.close()
tPltEnd=datetime.now()

print("plot time: ",tPltEnd-tPltStart)