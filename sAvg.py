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

beta=100
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

t=0.4
J=2.5
g=-0.05
inFilePrefix="beta"+str(beta)+"t"+str(t)+"J"+str(J)+"g"+str(g)
inPklFile=inFilePrefix+"out.pkl"

with open(inPklFile,"rb") as fptr:
    record=pickle.load(fptr)
tLoadEnd=datetime.now()
print("loading time: ",tLoadEnd-tLoadStart)

tMeanStart=datetime.now()

sMeanAll=[]
for i in range(0,len(record.sAll)):
    sMeanAll.append(np.mean(record.sAll[i]))

tMeanEnd=datetime.now()

print("avg time: ",tMeanEnd-tMeanStart)

plt.figure()
plt.plot(sMeanAll,color="black")
plt.xlabel("step")
plt.ylabel("$<s>$")

plt.savefig(inFilePrefix+"Avgs.png")
print(sMeanAll[-50:])