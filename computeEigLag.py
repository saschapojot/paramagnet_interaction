import pickle

from scipy.sparse import kron
from scipy.sparse import lil_matrix
import numpy as np
from scipy.sparse import eye
import random
from scipy.sparse import diags
# from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from datetime import datetime
from multiprocessing import Pool
from copy import deepcopy
import statsmodels.api as sm
#this script computes the eigenvalue problem of each element in the Markov chain
#for 1 set of [L,M,J,t,g] parameters

random.seed(100)
L=10
M=20

Ne=M


t=0.4
J=-2.5
g=0.05
KSupValsAll=[2*np.pi*j/(L*M) for j in range(0,M)]
T=0.01
procNum=48
#construct h(K,s)
# hPart=lil_matrix((2 * L, 2 * L), dtype=complex)
I2=eye(2,dtype=complex,format="lil")
#subdiagonals
midMat=lil_matrix((L,L),dtype=complex)

for j in range(1,L):
    midMat[j-1,j]=1
    midMat[j,j-1]=1
midMat*=-t
hPart=kron(midMat,I2)

#spin upup, spin downdown
upup=lil_matrix((2,2),dtype=complex)
upup[0,0]=1
downdown=lil_matrix((2,2),dtype=complex)
downdown[1,1]=1

#elem [L-1,0]
mat0=lil_matrix((L,L),dtype=complex)
mat0[L-1,0]=-t
mat0I2=kron(mat0,I2)
#elem [0,L-1]
mat1=lil_matrix((L,L),dtype=complex)
mat1[0,L-1]=-t
mat1I2=kron(mat1,I2)

JI2L=J*eye(2*L,dtype=complex,format="lil")

def hEig(js):
    """

    :param js: j: index of K, s: vector of magnetization on each site
    :return: j, eigenvalues and eigenvectors of matrix h
    """
    j,s=js

    K=KSupValsAll[j]

    h=hPart+np.exp(1j*L*K)*mat0I2+np.exp(-1j*L*K)*mat1I2

    sSum=0
    for l in range(0,L):
        sSum+=s[l]*s[(l+1)%L]
    h+=sSum*JI2L

    dg = g * diags(s, dtype=complex, format="lil")
    h += kron(dg, upup) - kron(dg, downdown)
    # tDiagStart=datetime.now()
    vals, vecs=eigh(h.toarray())
    # tDiagEnd=datetime.now()
    # print("diagonalization time: ",tDiagEnd-tDiagEnd)
    return [j,vals,vecs]

def bisection_method(f,tol=1e-8,maxiter=10000):
    """

    :param f: a monotonically increasing function
    :param tol: length of the interval containing the root
    :param maxiter: maximum iteration number
    :return: root of f=0
    """
    a=-1
    b=1
    leftSearchLenth=1
    rightSearchLength=1
    #search for left end
    while f(a)>0:
        a-=leftSearchLenth
        leftSearchLenth*=2
    # print("a="+str(a))
    #search for right end
    while f(b)<0:
        b+=rightSearchLength
        rightSearchLength*=2
    # print("b="+str(b))
    for _ in range(maxiter):
        midpoint = (a + b) / 2
        midVal=f(midpoint)
        # print("f(midpoint)="+str(midVal))
        if np.abs(midVal)<1e-16 or (b-a)/2<tol:
            return midpoint

        if f(a)*midVal<0:
            b=midpoint
        else:
            a=midpoint
    # print("_="+str(_))
    return (a+b)/2


def chemicalPotential(EVec):
    """

    :param EVec: a vector containing E for all K for all j=0,1,...,2L-1
    :return: chemical potential mu
    """
    EVec=np.array(EVec)

    def muf(mu):
        # sumTmp=np.sum(1/(np.exp(beta*(EVec-mu))+1))
        occ=[1/(np.exp(beta*(e-mu))+1) for e in EVec]
        sumTmp=np.sum(occ)
        return sumTmp-Ne
    retVal=bisection_method(muf)

    return retVal



def avgEnergy(EVec):
    """

    :param EVec: a vector containing E for all K for all j=0,1,...,2L-1
    :return: average value of energy, chemical potential
    """
    muVal=chemicalPotential(EVec)
    weightedEng=[1/(np.exp(beta*(e-muVal))+1)*e for e in EVec]

    return [np.sum(weightedEng), muVal]

def s2Eig(sCurr):
    """

    :param sCurr: current value of vector s
    :return: eigenvalues and eigenvectors given s
    """
    inValsAll=[[j,sCurr] for j in range(0,len(KSupValsAll))]
    pool0=Pool(procNum)
    retAll=pool0.map(hEig,inValsAll)
    return retAll
def s2EigSerial(sCurr):
    """

    :param sCurr: current value of vector s
    :return: eigenvalues and eigenvectors given s serially
    """
    retAll=[]
    for j in range(0,len(KSupValsAll)):
        retAll.append(hEig([j,sCurr]))
    retAllSorted=sorted(retAll,key=lambda item: item[0])
    return retAllSorted
def combineRetFromhEig(retAll):
    """

    :param retAll: results from function oneStepEvolution(sCurr)
    :return: combined values of eigenvalues
    """
    EVec=[]
    for item in retAll:
        _,vals,_=item
        for e in vals:
            EVec.append(e)
    return EVec



# sVals=[-1,1]
# sRealizations=[]
# for i in range(0,L):
#     sRealizations.append(sVals[random.randint(0,1)])
#
# tStart=datetime.now()
# retAll=oneStepEvolution(sRealizations)
# print(combineRetFromhEig(retAll))
# tEnd=datetime.now()
#
# print("one step time: ",tEnd-tStart)

# TEst=1000#equilibration time estimated

#TODO: estimate a more accurate value of TEst using time series
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
        self.chemPotAll=[]
        self.TEq=1000
        self.equilibrium=False

# def flipInd(i):
#     """
#
#     :return: random index to be flipped
#     """
#
#     return random.randint(0,L-1)

# def genUnif(i):
#     """
#
#     :param i:
#     :return: random number in [0,1
#     """
#     return random.random()

record=computationData()
# totalMCLength=2

#indices of s to be flippd
# pool1=Pool(procNum)
# indsFlipAll=pool1.map(flipInd,list(range(0,totalMCLength)))

#uniform distribution on [0,1)
# pool2=Pool(procNum)
# realUnif=pool2.map(genUnif,list(range(0,totalMCLength)))
# print(realUnif)
# print(indsAll)

#init s
tInitStart=datetime.now()
sVals=[-1,1]
sCurr=[]
for i in range(0,L):
    sCurr.append(sVals[random.randint(0,1)])
sCurr=np.array(sCurr)
# record.T=TEst

#init eigenvalues and eigenvectors
retAll=s2EigSerial(sCurr)
EVec=combineRetFromhEig(retAll)
EAvgCurr,muCurr=avgEnergy(EVec)
tInitEnd=datetime.now()
print("init time: ",tInitEnd-tInitStart)

tMCStart=datetime.now()


def autc(sAll):
    """

    :param sAll: contaning s vectors as rows
    :return: whether autocorrelation of each variable is low for long lags
    """
    lag=32
    num=10
    lastRowNums=lag*num
    minRowNum=1000+lastRowNums

    if len(sAll)<minRowNum:
        return False

    sAllLast=np.array(sAll[-lastRowNums::lag])


    colNum=len(sAllLast[0,:])
    # print(colNum)

    reachEq=True
    for i in range(0,colNum):
        acfi= sm.tsa.acf(sAllLast[:, i], nlags=lag)
        avgi=np.mean(np.abs(acfi))
        reachEq=reachEq and (avgi<1e-2)
    return reachEq


active=True
maxEquilbrationStep=10000

toEquilibriumCounter=0
tau=0
tEqStart=datetime.now()
flipNum=0
notFlipNum=0
print("T="+str(T))
beta=1/T
#to reach equilibrium of MCMC
while active:
    print("step "+str(tau))

    tOneMCStepStart=datetime.now()
    #flip s
    sNext = deepcopy(sCurr)
    flipIndVal=random.randint(0,L-1)
    sNext[flipIndVal] *= -1
    # tEigStart=datetime.now()
    retAllNext = s2EigSerial(sNext)
    # tEigEnd=datetime.now()
    # print("one step eig time: ",tEigEnd-tEigStart)
    # tSolveEqnStart=datetime.now()
    EVecNext = combineRetFromhEig(retAllNext)

    EAvgNext,muNext = avgEnergy(EVecNext)
    DeltaE = (EAvgNext - EAvgCurr)/M
    # tSolveEqnEnd=datetime.now()
    # print("solve mu :",tSolveEqnEnd-tSolveEqnStart)
    # tFlipStart=datetime.now()
    print("Delta E="+str(DeltaE))
    if DeltaE <= 0:
        sCurr = deepcopy(sNext)
        retAll = deepcopy(retAllNext)
        EAvgCurr=EAvgNext
        muCurr=muNext
        print("flipped")
        flipNum+=1
    else:
        r=random.random()

        print("r="+str(r))
        print("exp(-beta*Delta E)=" + str(np.exp(-beta * DeltaE)))
        if r < np.exp(-beta * DeltaE):
            sCurr = deepcopy(sNext)
            retAll = deepcopy(retAllNext)
            EAvgCurr = EAvgNext
            muCurr=muNext
            print("flipped")
            flipNum+=1
        else:
            print("not flipped")
            notFlipNum+=1
    # tFlipEnd=datetime.now()
    # print("flip time: ",tFlipEnd-tFlipStart)
    record.sAll.append(sCurr)
    record.EAvgAll.append(EAvgCurr)
    record.data.append(retAll)
    record.chemPotAll.append(muCurr)
    # EVecTmp = combineRetFromhEig(retAll)
    # EMax=np.max(EVecTmp)
    # EMin=np.min(EVecTmp)
    # print("Emin="+str(EMin))
    # print("mu="+str(muCurr))
    print("sCurr="+str(sCurr))
    # print("EMax="+str(EMax))
    tOneMCStepEnd=datetime.now()
    print("one step MC :",tOneMCStepEnd-tOneMCStepStart)
    tau+=1
    print("=====================================")

    if tau%500==0:
        print("flip "+str(tau))
    toEquilibriumCounter+=1
    if toEquilibriumCounter>maxEquilbrationStep:
        break
    # if tau>=5000 and  tau%1000==0:
    #     reachEq=autc(record.sAll)
    #     if reachEq==True:
    #         record.equilibrium=True
    #         active=False

tEqEnd=datetime.now()
print("equilibrium time: ",tEqEnd-tEqStart)
TEq=tau-1
record.TEq=TEq

tSampleStart=datetime.now()
#sampling after equilibrium
for tau in range(TEq,TEq+blkNum*blkSize):
    print("step " + str(tau))
    tOneMCStepStart = datetime.now()
    # flip s
    if tau%500==0:
        print("flip "+str(tau))
    sNext = deepcopy(sCurr)
    flipIndVal = random.randint(0, L - 1)
    sNext[flipIndVal] *= -1
    retAllNext = s2EigSerial(sNext)
    EVecNext = combineRetFromhEig(retAllNext)
    EAvgNext,muNext = avgEnergy(EVecNext)
    DeltaE = (EAvgNext - EAvgCurr)/M
    print("Delta E=" + str(DeltaE))
    if DeltaE <= 0:
        sCurr = deepcopy(sNext)
        retAll = deepcopy(retAllNext)
        EAvgCurr = EAvgNext
        muCurr=muNext
        print("flipped")
        flipNum+=1
    else:
        r = random.random()

        print("r=" + str(r))
        print("exp(-beta*Delta E)=" + str(np.exp(-beta * DeltaE)))
        if r < np.exp(-beta * DeltaE):
            sCurr = deepcopy(sNext)
            retAll = deepcopy(retAllNext)
            EAvgCurr = EAvgNext
            muCurr = muNext
            print("flipped")
            flipNum+=1
        else:
            print("not flipped")
            notFlipNum+=1

    record.sAll.append(deepcopy(sCurr))
    record.EAvgAll.append(EAvgCurr)
    record.data.append(deepcopy(retAll))
    record.chemPotAll.append(muCurr)
    # EVecTmp = combineRetFromhEig(retAll)
    # EMax = np.max(EVecTmp)
    # EMin = np.min(EVecTmp)
    # print("Emin=" + str(EMin))
    # print("mu=" + str(muCurr))
    # print("EMax=" + str(EMax))
    print("sCurr=" + str(sCurr))
    tOneMCStepEnd = datetime.now()
    print("one step MC :", tOneMCStepEnd - tOneMCStepStart)
    print("=====================================")



tSampleEnd=datetime.now()

print("Sampling time: ",tSampleEnd-tSampleStart)



tMCEnd=datetime.now()
print("MC time: ", tMCEnd-tMCStart)
print("flip num: "+str(flipNum))
print("no flip num: "+str(notFlipNum))
outPklFileName="T"+str(T)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"out.pkl"
with open(outPklFileName,"wb") as fptr:
    pickle.dump(record,fptr, pickle.HIGHEST_PROTOCOL)








































