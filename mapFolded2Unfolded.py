import numpy as np
from scipy.linalg import eigh
from multiprocessing import Pool
from datetime import datetime

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



def primEigenValsAndVecs(hprim,kPrimAll):
    """

    :param hprim: matrix function, (unperturbed) Hamiltonian of primitive cell
    :param kPrimAll: momentum values in BZ (for primitive cell)
    :return: [...,[n,eigvals,eigvecs],...], sorted by n
    """
    NPrim=len(kPrimAll)


    def oneEig(n):
        """

        :param n: index in kPrimIndsAll
        :return: eigenvalues and eigenvectors of hprim(kn)
        """
        hPrimMat=hprim(kPrimAll[n])
        vals,vecs=eigh(hPrimMat)
        return [n,vals,vecs]

    kPrimIndsAll = [i for i in range(0, NPrim)]

    procNum=48

    pool0=Pool(procNum)
    tPrimEigStart=datetime.now()
    retAll=pool0.map(oneEig,kPrimIndsAll)

    retSortedAll=sorted(retAll,key=lambda item:item[0])

    tPrimEigEnd=datetime.now()

    print("eig time for primitive cell Hamiltonian: ",tPrimEigEnd-tPrimEigStart)
    return retSortedAll



def constructAlly(retSortedAll,supCellLength,KSupValsAll):
    """

    :param retSortedAll: all eigenvectors and eigenvalues, from primEigenValsAndVecs(hprim,kPrimAll)
    :param supCellLength: length of supercell
    :param KSupValsAll: momentum values in SBZ (supercell)
    :return: all of the y vectors
    """
    M=len(KSupValsAll)
    if len(retSortedAll)%supCellLength !=0:
        print("invalid length")
        exit(1)

    sizehPrim=len(retSortedAll[0][1])

    yAllMat=np.zeros((M,supCellLength,sizehPrim,sizehPrim*supCellLength),dtype=complex)

    def oney(maj):
        """

        :param maj: maj=[m,a,j]
        :return: y_{j}^{(a)}(K_{m})
        """
        m,a,j=maj
        Km=KSupValsAll[m]
        n=m+a*M
        xVec=retSortedAll[n][2][:,j]

        yVec=x2y(a,supCellLength,Km,xVec)

        return [m,a,j,yVec]

    majAll=[[m,a,j] for m in range(0,M) for a in range(0,supCellLength) for j in range(0,sizehPrim)]

    procNum=48
    pool1=Pool(procNum)
    tyStart=datetime.now()
    retyAll=pool1.map(oney,majAll)

    for item in retyAll:
        m,a,j,yVec=item
        yAllMat[m,a,j,:]=yVec

    tyEnd=datetime.now()

    print("construct all y: ",tyEnd-tyStart)

    return yAllMat



def perturbedSupEigenValsAndVecs(hSupPerturbed,KSupValsAll):
    """
    :param hSupPerturbed: perturbed Hamiltonian for momentum in SBZ (supercell)
    :param KSupValsAll: momentum values in SBZ (supercell)
    :return: [...,[m,eigvals,eigvecs],...], sorted by m
    """
    M=len(KSupValsAll)

    KSupIndsAll=[i for i in range(0,M)]

    def oneEig(m):
        """

        :param m: index in KSupIndsAll
        :return: eigenvalues and eigenvectors of hSupPerturbed(Km)
        """
        Km=KSupValsAll[m]
        hSupPerturbedMat=hSupPerturbed(Km)
        vals,vecs=eigh(hSupPerturbedMat)
        return [m,vals,vecs]
    procNum=48

    pool2=Pool(procNum)
    tSupEigStart=datetime.now()
    retPertSupAll=pool2.map(oneEig,KSupIndsAll)
    retPertSupSortedAll=sorted(retPertSupAll,key=lambda item:item[0])
    tSupEigEnd=datetime.now()

    print("eig time for supercell perturbed Hamiltonian: ",tSupEigEnd-tSupEigStart)
    return  retPertSupSortedAll


def proj(yAllMat,retPertSupSortedAll):
    """

    :param yAllMat: all of the y vectors, constructed from (unperterbed) supercell Hamiltonian
    :param retPertSupSortedAll: all of the z vectors, solved from perturbed supercell Hamiltonian
    :return:
    """
    M, supCellLength, sizehPrim,_=yAllMat.shape

    def oneAMat(ma):
        """

        :param ma: [m,a]
        :return: all projections
        """
        m,a=ma
        A=[]
        for b in range(0,2*supCellLength):
            for j in range(0,sizehPrim):
                zTmp=retPertSupSortedAll[m][2][:,b]
                yTmp=yAllMat[m,a,j,:]
                cTmp=np.abs(np.vdot(zTmp,yTmp))
                ETmp=retPertSupSortedAll[m][1][b]
                oneRowTmp=[m,a,j,ETmp,cTmp]
                A.append(oneRowTmp)
        A=sorted(A,key=lambda row: row[3])#sort by the value of E
        return [m,a,A]

    allma=[[m,a] for m in range(0,M) for a in range(0,supCellLength)]

    procNum=48

    tAllAStart=datetime.now()
    pool3=Pool(procNum)
    retAllAMat=pool3.map(oneAMat,allma)

    tAllAEnd=datetime.now()

    print("all A mats time: ",tAllAEnd-tAllAStart)

    return retAllAMat



def mapping(hprim,kPrimAll,hSupPerturbed,KSupValsAll,supCellLength):
    """

    :param hprim: matrix function, (unperturbed) Hamiltonian of primitive cell
    :param kPrimAll: momentum values in BZ (for primitive cell)
    :param hSupPerturbed: perturbed Hamiltonian for momentum in SBZ (supercell)
    :param KSupValsAll: momentum values in SBZ (supercell)
    :param supCellLength: length of supercell
    :return:
    """

    #solve eigenvalue problem of (unperturbed) primitive cell's Hamiltonian
    retSortedAllPrim=primEigenValsAndVecs(hprim,kPrimAll)

    #construct (unperturbed) supercell Hamiltonian's eigenvectors
    yAllMat=constructAlly(retSortedAllPrim,supCellLength,KSupValsAll)

    #solve the eigenvalue problem of (perturbed) supercell's Hamiltonian
    retPertSupSortedAll=perturbedSupEigenValsAndVecs(hSupPerturbed,KSupValsAll)

    #perform projections to unfold the energy bands of the perturbed supercell's Hamiltonian
    retAllAMat=proj(yAllMat,retPertSupSortedAll)

    return retAllAMat