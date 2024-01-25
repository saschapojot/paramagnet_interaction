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
#This script computes magnetic susceptibility  over MC steps
# blkSize=100
# blkNum=50


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



TemperaturesAll=[1.7]
randSeedAll=[10,38,999,756,10992]



t=0.4
J=-2.5
g=0.05

part=3

sAvg=[]
chiValAll=[]
tPltStart=datetime.now()
inDir="./part"+str(part)+"/"
for T in TemperaturesAll:
    inFilePrefix =inDir+"T" + str(T) + "t" + str(t) + "J" + str(J) + "g" + str(g)
    inPklFile = inFilePrefix +"part"+str(part)+ "out.pkl"
    tLoadStart = datetime.now()
    with open(inPklFile, "rb") as fptr:
        record = pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("loading time: ", tLoadEnd - tLoadStart)
    sLast = record.sAll[-1000::10]
    # sAbs=np.abs(sLast)
    smTmp=np.mean(sLast,axis=1)#mean spin for each configuration
    # print(len(smTmp))
    sVal=np.mean(np.abs(smTmp))
    sAvg.append(sVal)

    ##average of spin over configurations
    meanS=np.mean(smTmp)
    #square of avg spin for one configuration
    sSquared=smTmp**2
    meanS2=np.mean((sSquared))
    chiTmp=(meanS2-meanS**2)/T
    chiValAll.append(chiTmp)



# phTransTemp=3
# indTr=TemperaturesAll.index(phTransTemp)
# sPhTr=sAvg[indTr]
#
# tempPrev=2
# indPrev=TemperaturesAll.index(tempPrev)
# sPrev=sAvg[indPrev]


fig,ax=plt.subplots()

ax.plot(TemperaturesAll,sAvg,color="black")
plt.xlabel("$T$")
plt.ylabel("<s>")
plt.title("Temperature from "+str(TemperaturesAll[0])+" to "+str(TemperaturesAll[-1]))
# Move left and bottom spines to zero
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# plt.vlines(x=phTransTemp,ymin=0,ymax=sPhTr,ls='--',color="red")
# plt.hlines(y=sPhTr,xmin=0,xmax=phTransTemp,ls="--",color="red")
# plt.vlines(x=tempPrev,ymin=0,ymax=sPrev,ls="--",color="blue")
# plt.hlines(y=sPrev,xmin=0,xmax=tempPrev,ls="--",color="blue")
xTicks=TemperaturesAll

plt.yticks([0,0.2,0.4,0.6,0.8,1])


ax.tick_params(axis='both', which='major', labelsize=6)
plt.xticks(xTicks)


plt.savefig(inDir+"T"+str(TemperaturesAll[0])+"toT"+str(TemperaturesAll[-1])+"sAvg.png")




plt.close()

fig,ax=plt.subplots()
ax.plot(TemperaturesAll,chiValAll,color="red")
plt.title("Temperature from "+str(TemperaturesAll[0])+" to "+str(TemperaturesAll[-1]))
plt.xlabel("$T$")
plt.ylabel("$\chi$")
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(axis='both', which='major', labelsize=6)
plt.savefig(inDir+"T"+str(TemperaturesAll[0])+"toT"+str(TemperaturesAll[-1])+"Chi.png")
plt.close()
tPltEnd=datetime.now()
print("time: ",tPltEnd-tPltStart)
