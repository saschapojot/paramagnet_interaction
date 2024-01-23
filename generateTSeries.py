import numpy as np
import re


lagFileName="computeEigLag"
suffix=".py"

part=2
TemperaturesAll=[0.1,0.2,0.3,0.4]+[0.5+n*0.5 for n in range(0,43)]


fileIn=open(lagFileName+suffix,"r")

contents=fileIn.readlines()
lineNum=0
for l in range(0,len(contents)):
    line=contents[l]
    if re.findall("^T=\d+",line):
        lineNum=l

contents[-3]='outPklFileName="T"+str(T)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"part"+str('+str(part)+')+"out.pkl"\n'


counter=0
for TVal in TemperaturesAll:
    contents[lineNum]="T="+str(TVal)+"\n"
    outFileName="computeEigLag"+str(counter)+"part"+str(part)+".py"

    fileOut=open(outFileName,"w+")
    for oneline in contents:
        fileOut.write(oneline)
    fileOut.close()
    counter+=1







for i in range(0,len(TemperaturesAll)):
    # TTmp=TemperaturesAll[i]
    bashContents = []
    bashContents.append("#!/bin/bash\n")
    bashContents.append("#SBATCH -n 12\n")
    bashContents.append("#SBATCH -N 1\n")
    bashContents.append("#SBATCH -p CLUSTER\n")
    bashContents.append("#SBATCH --mem=20GB\n")
    bashContents.append("#SBATCH -o outlag" + str(i) + ".o\n")
    bashContents.append("#SBATCH -e outlag" + str(i) + ".e\n")
    bashContents.append("cd /home/cywanag/liuxi/Documents/pyCode/paramagnet_interaction\n")
    bashContents.append("python3 computeEigLag"+str(i)+"part"+str(part)+".py > part"+str(part)+"rec"+str(i)+".txt\n")
    bsFileName="lag"+str(i)+".sh"
    fbsTmp=open(bsFileName,"w+")
    for oneline in bashContents:
        fbsTmp.write(oneline)
    fbsTmp.close()

