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



#this script loads data to check

L=10
M=20

Ne=L*M


t=0.4
J=2.5
g=-0.05
KSupValsAll=[2*np.pi*j/(L*M) for j in range(0,M)]
beta=10
blkSize=100
blkNum=50
procNum=48
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
inPklFileName="beta"+str(beta)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"out.pkl"

with open(inPklFileName,"rb") as fptr:
    record=pickle.load(fptr)
tLoadEnd=datetime.now()
print("loading time: ",tLoadEnd-tLoadStart)
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
    # tDiagStart=datetime.now()
    vals, vecs=eigh(h.toarray())
    # tDiagEnd=datetime.now()
    # print("diagonalization time: ",tDiagEnd-tDiagEnd)
    return [j,s,vals,vecs]


def bisection_method(f,tol=1e-9,maxiter=10000):
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
    :return: average value of energy
    """
    muVal=chemicalPotential(EVec)
    weightedEng=[1/(np.exp(beta*(e-muVal))+1)*e for e in EVec]

    return np.sum(weightedEng)


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
def s2EigSerial(sCurr):
    """

    :param sCurr: current value of vector s
    :return: eigenvalues and eigenvectors given s serially
    """
    retAll=[]
    for j in range(0,len(KSupValsAll)):
        retAll.append(hEig([j,sCurr]))
    return retAll
tau=860
sCurr=record.sAll[tau]
retCurr=record.data[tau]
EAvgCurr=record.EAvgAll[tau]
print(sCurr)
print("curr EAvg is "+str(EAvgCurr))

flipInd=2

sNext=deepcopy(sCurr)
sNext[flipInd]*=-1
print("flip position "+str(flipInd))
retAllNext = s2EigSerial(sNext)
EVecNext = combineRetFromhEig(retAllNext)
EAvgNext = avgEnergy(EVecNext)
print("next EAvg is "+str(EAvgNext))
DeltaE = EAvgNext - EAvgCurr
print("Delta E=" + str(DeltaE))

