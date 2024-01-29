import numpy as np
# from scipy.linalg import eigh
from multiprocessing import Pool
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import glob
import re
from pathlib import Path
#this script computes unfolded energy bands

class mapperUnfolding():
    def __init__(self, retEigPrimSortedAll,retEigSupSortedAll):
        """

        :param retEigPrimSortedAll: eigenvalues and eigenvectors solved from (unperturbed) primitive cell's Hamiltonian
        :param retEigSupSortedAll: eigenvalues and eigenvectors solved from (perturbed) supercell's Hamiltonian
        """
        retEigSupSortedAll=sorted(retEigSupSortedAll,key=lambda  item:item[0])
        retEigPrimSortedAll = sorted(retEigPrimSortedAll, key=lambda item: item[0])
        self.sizehPrim=len(retEigPrimSortedAll[0][1])# size of (unperturbed) primitive cells' Hamiltonian
        self.retEigPrimSortedAll=retEigPrimSortedAll
        self.retEigSupSortedAll=retEigSupSortedAll
        self.NPrim=len(self.retEigPrimSortedAll)#number of values in BZ (of primitive cell)
        self.M=len(self.retEigSupSortedAll)# number of values in SBZ (of supercell)
        self.supCellLength=int(self.NPrim/self.M)
        if self.NPrim%self.M !=0:
            print("incompatible lengths")
            exit(1)
        self.kPrimAll=[2*np.pi*n/self.NPrim for n in range(0,self.NPrim)] #values in BZ
        self.KSupValsAll=[2*np.pi*m/self.NPrim for m in range(0,self.M)]
        self.yAllMat = np.zeros((self.M, self.supCellLength, self.sizehPrim, self.sizehPrim * self.supCellLength), dtype=complex)




    def x2y(self,a,Km,xVec):
        """

        :param a: a=0,1,...,supCellLength-1, group number
        :param Km: momentum in SBZ
        :param xVec: eigenvector solved from (unperturbed) primitive cell
        :return: eigenvector of (unperturbed) supercell
        """
        yVec = []
        for r in range(0, self.supCellLength):
            yVec += list(xVec)
        yVec = np.array(yVec,dtype=complex)
        length = len(xVec)
        for r in range(0, self.supCellLength):
            yVec[r * length:(r + 1) * length] *= np.exp(1j * r * Km) * np.exp(1j * 2 * np.pi * r * a / self.supCellLength)
        yVec /= np.linalg.norm(yVec, ord=2)  # normalization
        return yVec

    def constructOney(self,maj):
        """

        :param maj: maj=[m,a,j]
        :return: y_{j}^{(a)}(K_{m})
        """
        m, a, j = maj
        Km = self.KSupValsAll[m]
        n = m + a * self.M
        xVec=self.retEigPrimSortedAll[n][2][:,j]
        yVec=self.x2y(a,Km,xVec)
        return [m,a,j,yVec]

    def constructAlly(self):
        """

        :return: all y
        """

        majAll = [[m, a, j] for m in range(0, self.M) for a in range(0, self.supCellLength) for j in
                       range(0, self.sizehPrim)]


        procNum=48
        pool1=Pool(procNum)
        # tyStart = datetime.now()
        retyAll=pool1.map(self.constructOney,majAll)

        for item in retyAll:
            m, a, j, yVec = item
            self.yAllMat[m, a, j, :] = yVec
        # tyEnd = datetime.now()
        # print("construct all y: ", tyEnd - tyStart)

    def oneAMat(self,ma):
        """

        :param ma: [m,a]
        :return: all projections for m and a
        """

        m,a=ma
        A=[]
        for b in range(0, self.sizehPrim * self.supCellLength):
            for j in range(0, self.sizehPrim):
                zTmp=self.retEigSupSortedAll[m][2][:,b]
                # zTmp/=np.linalg.norm(zTmp,ord=2)
                yTmp=self.yAllMat[m,a,j,:]
                cTmp=np.abs(np.vdot(zTmp,yTmp))
                ETmp=self.retEigSupSortedAll[m][1][b]
                oneRowTmp=[m,a,j,ETmp,cTmp,b]
                A.append(oneRowTmp)

        A = sorted(A, key=lambda row: row[3])  # sort by the value of E
        return [m, a, A]

    def proj(self):
        """

        :return: all of the projections
        """
        allma = [[m, a] for m in range(0, self.M) for a in range(0, self.supCellLength)]
        procNum=48

        # tAllAStart = datetime.now()
        pool3 = Pool(procNum)
        retAllAMat = pool3.map(self.oneAMat, allma)
        # tAllAEnd = datetime.now()
        # print("all A mats time: ", tAllAEnd - tAllAStart)
        self.retAllAMat=retAllAMat


    def mapping(self):
        """

        :return: unfolded band and weights
        """
        self.constructAlly()
        self.proj()

    def allA2EMatcMat(self):
        """

        :return: All of the unfolded E values and weights
        """
        nRow = len(self.retAllAMat[0][2])
        EMat = np.zeros((nRow, self.NPrim), dtype=float)
        cMat = np.zeros((nRow, self.NPrim), dtype=float)
        for item in self.retAllAMat:
            _, _, A = item
            for j in range(0, nRow):
                m, a, _, E, c, _ = A[j]
                n = m + a * M
                EMat[j, n] = E
                cMat[j, n] = c ** 2

        return EMat, cMat



class computationData:#holding computational results to be dumped using pickle
    def __init__(self):
        # self.T=TEst
        self.blkSize=1
        self.blkNum=1
        self.data=[]
        self.sAll=[]
        self.EAvgAll=[]
        self.chemPotAll = []
        self.TEq=1000
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

bandsDir=inDir+"/energyBands/"
Path(bandsDir).mkdir(parents=True, exist_ok=True)

L=10
M=20
NPrim=L*M
t=0.4
lastNum=20000#use the last lastNum configurations
separation=100#separation of the used configurations
kPrimAll=[2*np.pi*n/NPrim for n in range(0,NPrim)]
kPrimIndsAll=[2*n/NPrim for n in range(0,NPrim)]
retEigPrimSortedAll=[]
for n in range(0,NPrim):
    kn=kPrimAll[n]
    vals=[-2*t*np.cos(kn),-2*t*np.cos(kn)]
    vecs=np.eye(2)
    oneRow=[n,vals,vecs]
    retEigPrimSortedAll.append(oneRow)

tMapStart=datetime.now()
for i in range(0,len(pklFileNames)):
    with open(pklFileNames[i] ,"rb") as fptr:
        record = pickle.load(fptr)
    tLoadEnd = datetime.now()
    dataLast = record.data[-lastNum::separation]
    EMatsAll = []
    cMatsAll = []
    for oneRet in dataLast:
        mapper = mapperUnfolding(retEigPrimSortedAll, oneRet)
        mapper.mapping()
        EMat, cMat = mapper.allA2EMatcMat()
        EMatsAll.append(EMat)
        cMatsAll.append(cMat)
    Num = len(dataLast)
    EMatAvg = np.zeros(EMatsAll[0].shape)
    cMatAvg = np.zeros(cMatsAll[0].shape)
    for EMat in EMatsAll:
        EMatAvg += EMat
    EMatAvg /= Num
    for cMat in cMatsAll:
        cMatAvg += cMat
    cMatAvg /= Num
    plt.figure()
    for j in range(0, len(EMatAvg)):
        plt.scatter(kPrimIndsAll, EMatAvg[j, :], s=cMatAvg[j, :], color="red")
    plt.xlabel("$k/\pi$")
    plt.ylabel("$E$")
    # muAvg=np.mean(chemPotLast)
    # # print(chemPotLast)
    # plt.hlines(y=muAvg, xmin=kPrimIndsAll[0], xmax=kPrimIndsAll[-1], colors='blue', linestyles='-', lw=2, label='average chemical potential')
    T=TValsAll[i]
    J=JValsAll[i]
    g=gValsAll[i]

    plt.title("$T=$" + str(T) + ", $t=$" + str(t) + ", $J=$" + str(J) + ", $g=$" + str(g))
    plt.savefig(bandsDir+"rec"+str(i)+"T" + str(T) + "t" + str(t) + "J" + str(J) + "g" + str(g) + ".png")