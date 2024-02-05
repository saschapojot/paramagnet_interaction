import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import glob
import pickle
from pathlib import Path
from scipy.stats import ks_2samp
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
#discard the first R samples
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


#####################################################
# effective length
subSampleSpin=record.sAll[R:]
sMeanAbsAll=np.abs(np.mean(subSampleSpin,axis=1))
rho=autc(sMeanAbsAll,1)
A=(1+rho)/(1-rho)
A*=2# we increase the value of A for safety
A=int(A)

#  Kolmogorov-SmirNov test

# KS test for |<s>|
selectedsAll=sMeanAbsAll[0::A]
if len(selectedsAll)%2==1:
    selectedsAll=selectedsAll[1:]

lenSelected=len(selectedsAll)

sPart1=selectedsAll[:int(lenSelected/2)]
sPart2=selectedsAll[int(lenSelected/2):]

sD,spval=ks_2samp(sPart1,sPart2)
sD=round(sD,3)
spval=round(spval,3)
#cumulative average for |<s>|
cs=np.cumsum(selectedsAll)
csDenom=[i for i in range(1,len(cs)+1)]
cumusAvg=cs/csDenom
outsDir=inDir+"diagnostics/sCumAvg/"
Path(outsDir).mkdir(parents=True, exist_ok=True)

fig,ax=plt.subplots()
ax.plot(range(R,R+len(cumusAvg)),cumusAvg,color="black")
xsTicks=[R,R+len(cumusAvg)*1/4,R+len(cumusAvg)*2/4,R+len(cumusAvg)*3/4,R+len(cumusAvg)*4/4]
plt.axvline(x = R, color = 'r', ls=":")
# print(R)
ax.set_title("$T=$"+str(TTmp)+", $t=$"+str(tTmp)+", $J=$"+str(JTmp)+", $g=$"+str(gTmp))
ax.set_xlabel("MC step")
plt.xticks(xsTicks)
ax.set_ylabel("cumulative average $<s>$")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
txtsStr="$D=$"+str(sD)+", p val = "+str(spval)
ax.text(0.5, 0.95, txtsStr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
outsFilePrefix="sCumAvg"+inPklFileName[8:-4]
plt.savefig(outsDir+outsFilePrefix+".png")
plt.close()
# KS test for E
subSampleEAvg=record.EAvgAll[R:]
selectedEAvgAll=subSampleEAvg[0::A]
if len(selectedEAvgAll)%2==1:
    selectedEAvgAll=selectedEAvgAll[1:]

lenEAvgSelected=len(selectedEAvgAll)

EAvgPart1=selectedEAvgAll[:int(lenEAvgSelected/2)]
EAvgPart2=selectedEAvgAll[int(lenEAvgSelected/2):]
ED,Epval=ks_2samp(EAvgPart1,EAvgPart2)
ED=round(ED,3)
Epval=round(Epval,3)
#cumulative average for EAvg
cE=np.cumsum(selectedEAvgAll)
cEDenom=np.array([i for i in range(1,len(cE)+1)])
cumuEAvg=cE/cEDenom
xEAvgTicks=[R,R+len(cumuEAvg)*1/4,R+len(cumuEAvg)*2/4\
    ,R+len(cumuEAvg)*3/4,R+len(cumuEAvg)*4/4]

outEDir=inDir+"diagnostics/ECumAvg/"
Path(outEDir).mkdir(parents=True, exist_ok=True)
txtEAvgStr="$D=$"+str(ED)+", p val = "+str(Epval)
# print(txtEAvgStr)
fig,ax=plt.subplots()
ax.plot(range(R,R+len(cumuEAvg)),cumuEAvg,color="red")
plt.title("$T=$"+str(TTmp)+", $t=$"+str(tTmp)+", $J=$"+str(JTmp)+", $g=$"+str(gTmp))
plt.xlabel("MC step")
plt.ylabel("cumulative average $E$")
plt.xticks(xEAvgTicks)
plt.axvline(x = R, color = 'blue', ls=":")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.95, txtEAvgStr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
outEFilePrefix="ECumAvg"+inPklFileName[8:-4]
plt.savefig(outEDir+outEFilePrefix+".png")
plt.close()

######
# KS test for mu

subSampleMu=record.chemPotAll[R:]
selectedMuAll=subSampleMu[0::A]

if len(selectedMuAll)%2==1:
    selectedMuAll=selectedMuAll[1:]
lenMuSelected=len(selectedMuAll)

muPart1=selectedMuAll[:int(lenMuSelected/2)]
muPart2=selectedMuAll[int(lenMuSelected/2):]

muD,mupval=ks_2samp(muPart1,muPart2)
muD=round(muD,3)
mupval=round(mupval,3)

#cumulative average for mu
cmu=np.cumsum(selectedMuAll)
cmuDenom=np.array([i for i  in range(1,len(cmu)+1)])
cumumMu=cmu/cmuDenom
xMuTicks=[R,R+len(cumumMu)*1/4,R+len(cumumMu)*2/4\
    ,R+len(cumumMu)*3/4,R+len(cumumMu)*4/4]
outMuDir=inDir+"diagnostics/muCumAvg/"
Path(outMuDir).mkdir(parents=True, exist_ok=True)

fig,ax=plt.subplots()
ax.plot(range(R,R+len(cumuEAvg)),cumumMu,color="black")
plt.title("$T=$"+str(TTmp)+", $t=$"+str(tTmp)+", $J=$"+str(JTmp)+", $g=$"+str(gTmp))
plt.xlabel("MC step")
plt.ylabel("cumulative average $\mu$")
plt.xticks(xMuTicks)
plt.axvline(x = R, color = 'green', ls=":")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.95, txtEAvgStr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
outMuFilePrefix="MuCumAvg"+inPklFileName[8:-4]
plt.savefig(outMuDir+outMuFilePrefix+".png")
plt.close()
