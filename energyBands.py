import pickle
from copy import deepcopy
import numpy as np
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.sparse import kron
from scipy.sparse import lil_matrix
from scipy.sparse import eye
from scipy.sparse import diags
from scipy.linalg import eigh


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
        self.chemPotAll = []
        self.TEq=1000
        self.equilibrium=False


t=0.4
J=2.5
g=-0.05
L=10
M=20

Ne=M







#construct eigenvectors for (unperturbed primitive cell)
tAllxVecsStart=datetime.now()
NPrim=M*L# number of points in BZ for primitive cell

kPrimAll=[2*np.pi*n/NPrim for n in range(0,NPrim)]

retValsVecsPrim=[]
for n in range(0,NPrim):
    kn=kPrimAll[n]
    vals=[-2*t*np.cos(kn),-2*t*np.cos(kn)]
    vecs=np.eye(2,dtype=complex)
    oneRow=[n,vals,vecs]
    retValsVecsPrim.append(oneRow)

tAllxVecsEnd=datetime.now()

print("all x vecs time: ",tAllxVecsEnd-tAllxVecsStart)

KSupValsAll=[2*np.pi*j/(L*M) for j in range(0,M)]

def x2y(a,supCellLength,Km,xVec):
    """

    :param a: a=0,1,...,supCellLength-1, group number
    :param supCellLength: length of supercell
    :param Km: momentum in SBZ
    :param xVec: eigenvector solved from (unperturbed) primitive cell
    :return: eigenvector of (unperturbed) supercell
    """

    yVec=[]
    for r in range(0,supCellLength):
        yVec+=list(xVec)
    yVec=np.array(yVec)

    length = len(xVec)
    for r in range(0, supCellLength - 1):
        yVec[r * length:(r + 1) * length] *= np.exp(1j * r * Km) * np.exp(1j * 2 * np.pi * r * a / supCellLength)
    yVec /= np.linalg.norm(yVec, ord=2)#normalization
    return yVec
supCellLength=L
sizehPrim=2
def oney(maj):
    """

    :param maj: maj=[m,a,j]
    :return: y_{j}^{(a)}(K_{m})
    """
    m,a,j=maj
    Km=KSupValsAll[m]
    n=m+a*M
    xVec=retValsVecsPrim[n][2][:,j]

    yVec=x2y(a,L,Km,xVec)

    return [m,a,j,yVec]

majAll=[[m,a,j] for m in range(0,M) for a in range(0,supCellLength) for j in range(0,sizehPrim)]

procNum=48
pool1=Pool(procNum)
tyStart=datetime.now()
retyAll=pool1.map(oney,majAll)
yAllMat = np.zeros((M, supCellLength, sizehPrim, sizehPrim * supCellLength), dtype=complex)

for item in retyAll:
    m, a, j, yVec = item
    yAllMat[m, a, j, :] = yVec

tyEnd=datetime.now()

print("construct all y: ",tyEnd-tyStart)



