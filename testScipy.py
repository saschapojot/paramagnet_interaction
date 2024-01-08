from scipy.sparse import kron
from scipy.sparse import lil_matrix
import numpy as np
from scipy.sparse import eye
import random
from scipy.sparse import diags
# from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from datetime import datetime
import sys

L=1000
M=20

j=13
t=0.4
J=2.5
g=-0.87

K=2*np.pi*(j)/(L*M)

I2=eye(2,dtype=complex,format="lil")

h=lil_matrix((2*L,2*L),dtype=complex)

#elem [L-1,0]
mat0=lil_matrix((L,L),dtype=complex)
mat0[L-1,0]=-t*np.exp(1j*L*K)
#elem [0,L-1]
mat1=lil_matrix((L,L),dtype=complex)
mat1[0,L-1]=-t*np.exp(-1j*L*K)

h+=kron(mat0,I2)+kron(mat1,I2)


#subdiagonals
midMat=lil_matrix((L,L),dtype=complex)
for j in range(1,L):
    midMat[j-1,j]=1
    midMat[j,j-1]=1
midMat*=-t

h+=kron(midMat,I2)
sVals=[-1,1]
sRealizations=[]
for i in range(0,L):
    sRealizations.append(sVals[random.randint(0,1)])

sRlSum=0
for l in range(0,L):
    sRlSum+=sRealizations[l]*sRealizations[(l+1)%L]

h+=J*sRlSum*eye(L*2,dtype=complex,format="lil")

dg=g*diags(sRealizations,dtype=complex,format="lil")
upup=lil_matrix((2,2),dtype=complex)
upup[0,0]=1

downdown=lil_matrix((2,2),dtype=complex)
downdown[1,1]=1

h+=kron(dg,upup)-kron(dg,downdown)

tStart=datetime.now()

eigs,vals=eigh(h.toarray())

tEnd=datetime.now()

print("eig time: ",tEnd-tStart)