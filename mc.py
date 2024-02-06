# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:13:57 2017
Monte-Carlo simulation code for 2D hexagonal lattice (only nearest neighbor)
Ising model, Heisenberg Hamiltonian, and Metropolis algorithm
@author: Runzhang Xu and Yun Su
"""
import numpy as np
from random import random
from random import randint

T = 150   # [Kelvin temperature]
f = open('T'+str(T)+'.txt','w')

### parameter for this Monte-Carlo simulation
size = 50             # [system size]
spin = 1.5            # [spin of each site]
J1 = 0.00211          # [exhange parameter of nearest neighbor]
#J2 = 0.000649        # [exhange parameter of next-nearest neighbor]
#J3 = -0.000194       # [exhange parameter of third-nearest neighbor]
k = 8.6173303e-5      # <Boltzmann Constant>
MC = 4000000*size**2  # [total Monte-Carlo steps]
### initializing sample lattice
s = np.zeros((size+2,size+2))
for a in range(size):
    for b in range(size):
        s[a+1,b+1] = (2*randint(1,2)-3)*spin
### periodic boundary condition
s[:,0] = s[:,size]
s[:,size+1] = s[:,1]
s[0,:] = s[size,:]
s[size+1,:] = s[1,:]
### initializing output variables
S = 0                 # spin sum over all MC steps
S2 = 0                # spin^2 sum over all MC steps
spin_sum = 0          # spin sum per MC step
E = 0                 # total energy sum over all MC steps
E2 = 0                # total energy^2 sum over all MC steps
E_site = 0            # Energy per site in lattice
### calculate total energy of sample lattice
for c1 in range(size):
    for c2 in range(size):
        if (abs(c1-c2)%2)==1:
            # sublattice A
            E_site = E_site - 0.5*s[c1+1,c2+1]*J1*(s[c1+1,c2]+s[c1+1,c2+2]+s[c1,c2+1])
        else:
            # sublattice B
            E_site = E_site - 0.5*s[c1+1,c2+1]*J1*(s[c1+1,c2]+s[c1+1,c2+2]+s[c1+2,c2+1])

mc = 0    # Monte-Carlo step count
ct = 0    # count number for average
### Main starts
while mc < MC:
    i = randint(1,size)  # pick site ramdomly
    j = randint(1,size)  # pick site ramdomly
    
	### energy difference after spin flip
    if (abs(i-j)%2)==1:
        # sublattice A
        E_diff = s[i,j]*J1*(s[i,j-1]+s[i,j+1]+s[i-1,j])
    else:
        # sublattice B
        E_diff = s[i,j]*J1*(s[i,j-1]+s[i,j+1]+s[i+1,j])
    
	### criteria for spin flip
    if E_diff <= 0:
        s[i,j] = -s[i,j]
        E_site = E_site + E_diff
    elif random() < np.exp(-E_diff/T/k):
        s[i,j] = -s[i,j]
        E_site = E_site + E_diff
    
	### update periodic boundary
    s[:,0] = s[:,size]
    s[:,size+1] = s[:,1]
    s[0,:] = s[size,:]
    s[size+1,:] = s[1,:]

    ### criteria for a measurement (start after sufficient annealing)
    if (mc > MC*0.8):
        mag_sum = abs(np.sum(s[1:size+1,1:size+1]))
        
        S = S + mag_sum/size**2            # spin sum over all MC steps
        S2 = S2 + mag_sum**2/size**2       # spin^2 sum over all MC steps
        E = E + E_site                     # total energy sum over all MC steps
        E2 = E2 + E_site**2                # total energy^2 sum over all MC steps
        
        ct = ct + 1                        # count number for average +1
    
#    f.write(str(mc)+',  '+str(S)+'\n')
    mc = mc + 1  # Monte-Carlo step count +1
### Main ends
### Date post-process
spin_site = S/ct                           # spin per site
X = (S2/ct - (S/ct)**2)/k/T                # magnetic susceptibility
C = (E2/ct - (E/ct)**2)/k/(T**2)           # specific heat

f.write(str(spin_site)+',  '+str(X)+',  '+str(C)+'\n')
f.close()
