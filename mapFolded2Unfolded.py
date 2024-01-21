import numpy as np
from scipy.linalg import eigh
from multiprocessing import Pool
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
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



blkSize=100
blkNum=50

beta=0.1

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
L=10
M=20
t=0.4
J=2.5
g=-0.05
inPklFile="beta"+str(beta)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"out.pkl"

with open(inPklFile,"rb") as fptr:
    record=pickle.load(fptr)
tLoadEnd=datetime.now()
print("loading time: ",tLoadEnd-tLoadStart)


NPrim=L*M

kPrimAll=[2*np.pi*n/NPrim for n in range(0,NPrim)]

retEigPrimSortedAll=[]
for n in range(0,NPrim):
    kn=kPrimAll[n]
    vals=[-2*t*np.cos(kn),-2*t*np.cos(kn)]
    vecs=np.eye(2)
    oneRow=[n,vals,vecs]
    retEigPrimSortedAll.append(oneRow)

dataLast=record.data[-500::10]

EMatsAll=[]
cMatsAll=[]

tAvgStart=datetime.now()
for oneRet in dataLast:
    mapper=mapperUnfolding(retEigPrimSortedAll,oneRet)
    mapper.mapping()

    EMat, cMat = mapper.allA2EMatcMat()
    EMatsAll.append(EMat)

    cMatsAll.append(cMat)

Num=len(dataLast)
EMatAvg=np.zeros(EMatsAll[0].shape)
cMatAvg=np.zeros(cMatsAll[0].shape)
for EMat in EMatsAll:
    EMatAvg+=EMat
EMatAvg/=Num

for cMat in cMatsAll:
    cMatAvg+=cMat
cMatAvg/=Num
tAvgEnd=datetime.now()
print("avg time: ",tAvgEnd-tAvgStart)
kPrimIndsAll=[2*n/NPrim for n in range(0,NPrim)]
plt.figure()
for j in range(0,len(EMatAvg)):
    plt.scatter(kPrimIndsAll,EMatAvg[j,:],s=cMatAvg[j,:],color="red")
plt.xlabel("$k/\pi$")
plt.ylabel("$E$")
plt.title("$\\beta=$"+str(beta)+", $t=$"+str(t)+", $J=$"+str(J)+", $g=$"+str(g))
plt.savefig("beta"+str(beta)+"t"+str(t)+"J"+str(J)+"g"+str(g)+".png")

# mapper=mapperUnfolding(retEigPrimSortedAll,record.data[45])
# mapper.mapping()
# retAllAMat=mapper.retAllAMat
# EMat, cMat=mapper.allA2EMatcMat()
#
#
# kPrimIndsAll=[2*n/NPrim for n in range(0,NPrim)]
# plt.figure()
# for j in range(0,len(EMat)):
#     plt.scatter(kPrimIndsAll,EMat[j,:],s=cMat[j,:],color="red")
#
# plt.savefig("tmp.png")


# m=2
# a=3
# counter=0
# for item in retAllAMat:
#     m_,a_,A_=item
#     if m_!=m or a_!=a:
#         continue
#     else:
#         for oneRow in A_:
#             _,_,_,E,_,_=oneRow
#             print(E)
#             counter+=1
# print(counter)
# counter=0
# setm=2
# setb=8
# projs0=[]
# projs1=[]
# tmp=0
# print(len(retAllAMat))
# for item in retAllAMat:
#     m,a,A=item
#     if m!=setm:
#         continue
#     else:
#         for oneRow in A:
#             _,_,j,E,c,b=oneRow
#             if b!=setb:
#                 continue
#             else:
#                 if j==0:
#                     projs0.append(c)
#                 else:
#                     projs1.append(c)
#
# projs0=np.array(projs0)
# projs1=np.array(projs1)
#
# tmp0=np.sum(projs0**2)
# tmp1=np.sum(projs1**2)
#
# print(tmp0)
# print(tmp1)


# print(sorted(inds,key=lambda item: item[0]))



#
# plt.figure()
#
# for item in retAllAMat:
#     m,a,A=item
#     for row in A:
#         kPrimTmp=2*(m+a*M)/NPrim
#         _,_,_,E,c=row
#         plt.scatter(2*(m+a*M)/NPrim,E,s=c**2*10)
#
# plt.savefig("tmp.png")
# plt.close()