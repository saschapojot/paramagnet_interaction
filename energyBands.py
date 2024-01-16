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
        self.TEq=1000
        self.equilibrium=False

tLoadStart=datetime.now()


inPklFile="./lagbeta"+str(beta)+"out.pkl"

with open(inPklFile,"rb") as fptr:
    record=pickle.load(fptr)
tLoadEnd=datetime.now()
print("loading time: ",tLoadEnd-tLoadStart)

lastNum=blkNum*blkSize*5
tryBlkMultiple=3

blkSizeNew=blkSize*tryBlkMultiple
sToBeUsed=np.array(record.sAll[-blkSizeNew*blkNum:])

_,nCol=sToBeUsed.shape

sblkAvg=np.zeros((blkNum,nCol),dtype=float)
for i in range (0,blkNum):
    tmp=sToBeUsed[i*blkSizeNew:(i+1)*blkSizeNew,:]
    sblkAvg[i,:]=tmp.mean(axis=0)

Nbt=100000
resampledRowInds=np.random.choice(list(range(0,blkNum)),Nbt,replace=True)


sResampled=np.zeros((Nbt,nCol),dtype=float)

for j in range(0,Nbt):
    indj=resampledRowInds[j]
    sResampled[j,:]=deepcopy(sblkAvg[indj,:])

S=sResampled.mean(axis=0)
print(S)
L=10
M=20

Ne=L*M
procNum=48
j=13
t=0.4
J=2.5
g=-0.87
KSupValsAll=[2*np.pi*j/(L*M) for j in range(0,M)]

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


tS2EigStart=datetime.now()
retKAll=s2Eig(S)
EVec=combineRetFromhEig(retKAll)
chemPot=chemicalPotential(EVec)
# EAvgCurr=avgEnergy(EVec)
tS2EigEnd=datetime.now()

print("eig time: ",tS2EigEnd-tS2EigStart)


def x2y(a,K,xvec):
    """

    :param a: group number, a=0,1,...,L-1
    :param K: momentum in SBZ
    :param xvec: eigenvector solved from primitive cell
    :return: map x eigenvector to y eigenvector
    """
    yvec=[]
    for r in range(0,L):
        yvec+=list(xvec)
    yvec=np.array(yvec)
    length=len(xvec)
    for r in range(0,L-1):
        yvec[r*length:(r+1)*length]*=np.exp(1j*r*K)*np.exp(1j*2*np.pi*r*a/L)
    return yvec

retKAllSorted=sorted(retKAll,key=lambda  item: item[0])


#eigenvectors for primitive cell
up=np.array([1,0])
down=np.array([0,1])
updown=np.eye(2,dtype=complex)
supKEigsVecs=[]##each entry of supKEigsVecs corresponds to one K in SBZ, unperturbed
for n in range(0,len(KSupValsAll)):
    K0=KSupValsAll[n]
    oneK=[]
    for a in range(0,L):
        yaMat=np.zeros((L*2,2),dtype=complex)
        for j in range(0,2):
            yaMat[:,j]=x2y(a,K0,updown[:,j])
        kp=K0+2*np.pi*a/L
        rowa=[a,n,[-2*t*np.cos(kp),-2*t*np.cos(kp)],yaMat]
        oneK.append(rowa)
    supKEigsVecs.append(oneK)


def zProjRow(z,row):
    """

    :param z: eigenvector of h(K,S)
    :param row: [a,n,vals,vecs]
    :return: abs of projection of z to each column vector in row
    """
    _,_,_,vecs=row
    coefs=[]
    rn,cn=vecs.shape
    zNormalized=z/np.linalg.norm(z,ord=2)
    for j in range(0,cn):
        vecTmpNormalized=vecs[:,j]/np.linalg.norm(vecs[:,j],ord=2)
        coefs.append(np.abs(np.vdot(zNormalized,vecTmpNormalized)))
    return np.array(coefs)



def projection(n):
    """

    :param n: index of K in SBZ
    :return: [n, [j,[inds0],[coefs],[inds1],[coefs],[inds2],[coefs],...]], j-th eigenvector of hs projected to each of the eigenvectors of h0s,
    inds0, inds1, inds2 are indices of non-zero projections
    """
    eps = 1e-2
    _,_,valsSK,vecsSK=retKAllSorted[n]#with S
    # row0, row1, row2 = supKEigsVecs[n]  # original
    oneK=supKEigsVecs[n]
    projInd=[n,[]]
    for j in range(0,len(valsSK)):
        jInd=[j]
        zTmp=vecsSK[:,j]
        for a in range(0,L):
            rowa=oneK[a]
            coefsaTmp=zProjRow(zTmp,rowa)
            indsaTmp=np.where(coefsaTmp>eps)[0]
            jInd.append(indsaTmp)
            jInd.append(coefsaTmp[indsaTmp])

        projInd[-1].append(jInd)
    return projInd

tProjStart=datetime.now()
pool1=Pool(procNum)
supKIndAll=list(range(0,len(KSupValsAll)))
retProjs=pool1.map(projection,supKIndAll)
tProjEnd=datetime.now()

print("projection time: ",tProjEnd-tProjStart)

sortedProjs=sorted(retProjs,key=lambda item: item[0])

knPlt=[[] for a in range(0,L)]
EPlt=[[] for a in range(0,L)]
sPlt=[[] for a in range(0,L)]

for item in sortedProjs:
    n,listTmp=item
    start=1
    for elem in listTmp:
        j=elem[0]
        for a in range(0,L):
            indsa=elem[start+2*a]
            coefsa=elem[start+1+2*a]
            for l in range(0,len(indsa)):
                na=n+a*M
                knPlt[a].append(na)
                EPlt[a].append(retKAllSorted[n][2][j])
                sPlt[a].append(coefsa[l])


from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, L))

multiply=10
plt.figure()
for i in range(0,L):
    plt.scatter(2*np.array(knPlt[i])/(M*L),EPlt[i],color="red",s=np.array(sPlt[i])*multiply)

plt.xlabel("$k/\pi$")
plt.ylabel("Energy")
plt.axhline(y=chemPot, color='blue', linestyle='-')
kPrimValsAll=[2*np.pi*n/(L*M) for n in range(0,L*M-1)]
kPrimValsAll=np.array(kPrimValsAll)
# plt.plot(kPrimValsAll/np.pi,-2*t*np.cos(kPrimValsAll),color="black")
plt.title("$\\beta=$"+str(beta))
plt.savefig("beta"+str(beta)+"unfolded.png")
plt.close()

