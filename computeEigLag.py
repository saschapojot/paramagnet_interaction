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


L=10
M=20

Ne=L*M

j=13
t=0.4
J=2.5
g=-0.87
KSupValsAll=[2*np.pi*j/(L*M) for j in range(0,M)]
beta=10
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
    :return: j, s, eigenvalues and eigenvectors of matrix h
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

    vals, vecs=eigh(h.toarray())

    return [j,s,vals,vecs]

def bisection_method(f,tol=1e-16,maxiter=10000):
    """

    :param f: an monotonically increasing function
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
    :return: average value of energy
    """
    muVal=chemicalPotential(EVec)
    weightedEng=[1/(np.exp(beta*(e-muVal))+1)*e for e in EVec]

    return np.sum(weightedEng)

def s2Eig(sCurr):
    """

    :param sCurr: current value of vector s
    :return: eigenvalues and eigenvectors given s
    """
    inValsAll=[[j,sCurr] for j in range(0,len(KSupValsAll))]
    pool0=Pool(procNum)
    retAll=pool0.map(hEig,inValsAll)
    return retAll

def combineRetFromhEig(retAll):
    """

    :param retAll: results from function oneStepEvolution(sCurr)
    :return: combined values of eigenvalues
    """
    EVec=[]
    for item in retAll:
        _,_,vals,_=item
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
retAll=s2Eig(sCurr)
EVec=combineRetFromhEig(retAll)
EAvgCurr=avgEnergy(EVec)
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
maxEquilbrationStep=100000

toEquilibriumCounter=0
tau=0
tEqStart=datetime.now()
#to reach equilibrium of MCMC
while active:
    #flip s
    sNext = deepcopy(sCurr)
    flipIndVal=random.randint(0,L-1)
    sNext[flipIndVal] *= -1
    retAllNext = s2Eig(sNext)
    EVecNext = combineRetFromhEig(retAllNext)
    EAvgNext = avgEnergy(EVecNext)
    DeltaE = EAvgNext - EAvgCurr
    if DeltaE <= 0:
        sCurr = deepcopy(sNext)
        retAll = deepcopy(retAllNext)
        EAvgCurr=EAvgNext
    else:
        if random.random() < np.exp(-beta * DeltaE):
            sCurr = deepcopy(sNext)
            retAll = deepcopy(retAllNext)
            EAvgCurr = EAvgNext

    record.sAll.append(deepcopy(sCurr))
    record.EAvgAll.append(EAvgCurr)
    record.data.append(deepcopy(retAll))
    tau+=1
    if tau%500==0:
        print("sweep "+str(tau))
    toEquilibriumCounter+=1
    if toEquilibriumCounter>maxEquilbrationStep:
        break
    if tau>=5000 and  tau%1000==0:
        reachEq=autc(record.sAll)
        if reachEq==True:
            record.equilibrium=True
            active=False

tEqEnd=datetime.now()
print("equilibrium time: ",tEqEnd-tEqStart)
TEq=tau-1
record.TEq=TEq

tSampleStart=datetime.now()
#sampling after equilibrium
for tau in range(TEq,TEq+blkNum*blkSize):
    # flip s
    if tau%500==0:
        print("sweep "+str(tau))
    sNext = deepcopy(sCurr)
    flipIndVal = random.randint(0, L - 1)
    sNext[flipIndVal] *= -1
    retAllNext = s2Eig(sNext)
    EVecNext = combineRetFromhEig(retAllNext)
    EAvgNext = avgEnergy(EVecNext)
    DeltaE = EAvgNext - EAvgCurr
    if DeltaE <= 0:
        sCurr = deepcopy(sNext)
        retAll = deepcopy(retAllNext)
        EAvgCurr = EAvgNext
    else:
        if random.random() < np.exp(-beta * DeltaE):
            sCurr = deepcopy(sNext)
            retAll = deepcopy(retAllNext)
            EAvgCurr = EAvgNext

    record.sAll.append(deepcopy(sCurr))
    record.EAvgAll.append(EAvgCurr)
    record.data.append(deepcopy(retAll))



tSampleEnd=datetime.now()

print("Sampling time: ",tSampleEnd-tSampleStart)



tMCEnd=datetime.now()
print("MC time: ", tMCEnd-tMCStart)

outPklFileName="beta"+str(beta)+"out.pkl"
with open(outPklFileName,"wb") as fptr:
    pickle.dump(record,fptr, pickle.HIGHEST_PROTOCOL)








































