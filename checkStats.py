import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import glob
import pickle
from pathlib import Path
#check stats of computed MC results
#we check the following:
#1. cumulative average
#2. effective length
#3. autocorrelation
#4. Kolmogorov-SmirNov test

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




part=3
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
    matchg=re.search(r"g(-?\d+(\.\d+)?)part",file)
    if matchg:
        gValsAll.append(matchg.group(1))


val0=(len(TValsAll)-len(tValsAll))**2\
    +(len(TValsAll)-len(tValsAll))**2\
    +(len(TValsAll)-len(JValsAll))**2\
    +(len(TValsAll)-len(gValsAll))**2\
    +(len(TValsAll)-len(pklFileNames))**2
# print("len(TValsAll)="+str(len(TValsAll)))
# print("len(tValsAll)="+str(len(tValsAll)))
# print("len(JValsAll)="+str(len(JValsAll)))
# print("len(gValsAll)="+str(len(gValsAll)))
# print("len(pklFileNames)="+str(len(pklFileNames)))


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


ldInd=-3
inPklFileName=pklFileNames[ldInd]
TTmp=TValsAll[ldInd]
tTmp=tValsAll[ldInd]
JTmp=JValsAll[ldInd]
gTmp=gValsAll[ldInd]
tLoadStart = datetime.now()
with open(inPklFileName, "rb") as fptr:
    record=pickle.load(fptr)
tLoadEnd = datetime.now()
print("loading time: ", tLoadEnd - tLoadStart)
R=int(len(record.sAll)/3)
def makeROdd(R):
    #if R is odd
    if R%2==1:
        return R
    else:
        return R+1
def makeREven(R):
    #if R is odd
    if R%2==1:
        return R+1
    else:
        return R
lenTot=len(record.sAll)
#if total length is odd
if lenTot%2==1:
    R=makeROdd(R)
else:
    R=makeREven(R)
###################################################
#1. (1)cumulative average of s
subSampleSpin=record.sAll[R:]

lenSubSample=len(subSampleSpin)

sMeanAbsAll=np.abs(np.mean(subSampleSpin,axis=1))

cs=np.cumsum(sMeanAbsAll)
csDenom=[i for i in range(1,len(cs)+1)]
cumusAvg=cs/csDenom

outsDir=inDir+"diagnostics/sCumAvg/"
Path(outsDir).mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(range(R,R+len(cumusAvg)),cumusAvg,color="black")
plt.title("$T=$"+str(TTmp)+", $t=$"+str(tTmp)+", $J=$"+str(JTmp)+", $g=$"+str(gTmp))
plt.xlabel("MC step")
plt.ylabel("cumulative average $<s>$")
outsFilePrefix="sCumAvg"+inPklFileName[8:-4]
plt.savefig(outsDir+outsFilePrefix+".png")
#1. (2) cumulative average of E
subSampleEAvg=record.EAvgAll[R:]
cE=np.cumsum(subSampleEAvg)
cEDenom=np.array([i for i in range(1,len(cE)+1)])
cumuEAvg=cE/cEDenom
outEDir=inDir+"diagnostics/ECumAvg/"
Path(outEDir).mkdir(parents=True, exist_ok=True)

plt.figure()
plt.plot(range(R,R+len(cumuEAvg)),cumuEAvg,color="red")
plt.title("$T=$"+str(TTmp)+", $t=$"+str(tTmp)+", $J=$"+str(JTmp)+", $g=$"+str(gTmp))
plt.xlabel("MC step")
plt.ylabel("cumulative average $E$")
outEFilePrefix="ECumAvg"+inPklFileName[8:-4]
plt.savefig(outEDir+outEFilePrefix+".png")

#1. (2) cumulative average of mu

subSampleMu=record.chemPotAll[R:]
cMu=np.cumsum(subSampleMu)
cMuDenom=np.array([i for i in range(1,len(cE)+1)])
cumuMuAvg=cMu/cMuDenom
outMuDir=inDir+"diagnostics/muCumAvg/"
Path(outMuDir).mkdir(parents=True, exist_ok=True)
plt.figure()
plt.plot(range(R,R+len(cumuMuAvg)),cumuMuAvg,color="blue")
plt.title("$T=$"+str(TTmp)+", $t=$"+str(tTmp)+", $J=$"+str(JTmp)+", $g=$"+str(gTmp))
plt.xlabel("MC step")
plt.ylabel("cumulative average $\mu$")
outmuFilePrefix="muCumAvg"+inPklFileName[8:-4]
plt.savefig(outMuDir+outmuFilePrefix+".png")
###############################################################

################################################################
#2. effective length

#using |<s>|
# sA=0
# for tau in range(1,len(sMeanAbsAll)):
#     sA+=autc(sMeanAbsAll,tau)
# sA*=2
# sA+=1
# print(sA)
rho=autc(sMeanAbsAll,1)

print((1+rho)/(1-rho))
