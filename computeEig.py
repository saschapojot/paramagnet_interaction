from scipy.sparse import kron
from scipy.sparse import lil_matrix
import numpy as np
from scipy.sparse import eye
import random
from scipy.sparse import diags
# from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from datetime import datetime

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
        print("f(midpoint)="+str(midVal))
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