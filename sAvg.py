import pickle
# from copy import deepcopy
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import glob
import re


#This script computes avgerage value of s
#This script computes magnetic susceptibility
#This script computes specific heat


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
# print("T vals = "+str(TValsAll))
# print("t vals = "+str(tValsAll))
# print("J vals = "+str(JValsAll))
# print("g vals = "+str(gValsAll))

print("T val length = "+str(len(TValsAll)))
print("t val length = "+str(len(tValsAll)))
print("J val length = "+str(len(JValsAll)))
print("g val length = "+str(len(gValsAll)))
print("file val length = "+str(len(pklFileNames)))

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

tPltStart=datetime.now()

sAvgAll=[]
chiValAll=[]
specificHeatAll=[]
lastNum=20000#use the last lastNum configurations
separation=100#separation of the used configurations

for i in range(0,len(pklFileNames)):
    inPklFileName=pklFileNames[i]
    tLoadStart = datetime.now()
    with open(inPklFileName, "rb") as fptr:
        record=pickle.load(fptr)
    tLoadEnd = datetime.now()
    print("loading time: ", tLoadEnd - tLoadStart)
    sLast = record.sAll[-lastNum::separation]
    smTmp = np.mean(sLast, axis=1)  # mean spin for each configuration
    sVal = np.mean(np.abs(smTmp))
    sAvgAll.append(sVal)
    ##average of spin over configurations
    meanS = np.mean(smTmp)
    # square of avg spin for one configuration
    sSquared = smTmp ** 2
    meanS2 = np.mean(sSquared)
    T=TValsAll[i]
    chiTmp = (meanS2 - meanS ** 2) / T
    chiValAll.append(chiTmp)

    #specific heat
    EAvgLast=np.array(record.EAvgAll[-lastNum::separation])
    meanE=np.mean(EAvgLast)
    EAvgLast2=EAvgLast**2
    meanE2=np.mean(EAvgLast2)
    CTmp=(meanE2-meanE**2)/T**2
    specificHeatAll.append(CTmp)

#plot <s> vs T
fig,ax=plt.subplots()
ax.plot(TValsAll,sAvgAll,color="black")
plt.xlabel("$T$")
plt.ylabel("<s>")
plt.title("Temperature from "+str(TValsAll[0])+" to "+str(TValsAll[-1]))

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.yticks([0,0.2,0.4,0.6,0.8,1])
ax.tick_params(axis='both', which='major', labelsize=6)
plt.savefig(inDir+"T"+str(TValsAll[0])+"toT"+str(TValsAll[-1])+"sAvg.png")
plt.close()





# print("chi:"+str(chiValAll))
# plot chi vs T
fig,ax=plt.subplots()
ax.plot(TValsAll,chiValAll,color="red")
plt.title("Temperature from "+str(TValsAll[0])+" to "+str(TValsAll[-1]))
plt.xlabel("$T$")
plt.ylabel("$\chi$")
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(axis='both', which='major', labelsize=6)
plt.savefig(inDir+"T"+str(TValsAll[0])+"toT"+str(TValsAll[-1])+"Chi.png")
plt.close()




#plot C vs T
fig,ax=plt.subplots()
ax.plot(TValsAll,specificHeatAll,color="blue")
plt.title("Temperature from "+str(TValsAll[0])+" to "+str(TValsAll[-1]))
plt.xlabel("$T$")
plt.ylabel("$C$")
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

# Hide top and right spines
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.tick_params(axis='both', which='major', labelsize=6)
plt.savefig(inDir+"T"+str(TValsAll[0])+"toT"+str(TValsAll[-1])+"specificHeat.png")



tPltEnd=datetime.now()
print("time: ",tPltEnd-tPltStart)


