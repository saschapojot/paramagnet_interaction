import numpy as np
import re


lagFileName="computeEigLag"
suffix=".py"

part=4
TemperaturesAll=[1.7]
randSeedAll=[10,38,999,756,10992]


fileIn=open(lagFileName+suffix,"r")

contents=fileIn.readlines()
lineTemperature=0#the line corresponding to T=xxxxx (temperature)
lineRandSeed=0# random seed
lineMaxEq=0# loop numbers in first mc
for l in range(0,len(contents)):
    line=contents[l]
    if re.findall("^T=\d+",line):
        lineTemperature=l
        # print(lineTemperature)
    if re.findall("^random\.seed",line):
        lineRandSeed=l
        # print(lineRandSeed)
    if re.findall("^maxEquilbrationStep=",line):
        lineMaxEq=l
        # print(lineMaxEq)





# contents[-3]='outPklFileName="T"+str(T)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"part"+str('+str(part)+')+"out.pkl"\n'
#
#
counter=0
for TVal in TemperaturesAll:
    for rs in randSeedAll:
        contents[lineTemperature]="T="+str(TVal)+"\n"
        contents[lineRandSeed]="random.seed("+str(rs)+")\n"
        outFileName = "computeEigLag" + str(counter) + "part" + str(part)+"randseed"+str(rs) + ".py"
        fileOut = open(outFileName, "w+")
        for oneline in contents:
            fileOut.write(oneline)
        fileOut.close()
        counter += 1


counter=0
for TVal in TemperaturesAll:
    for rs in randSeedAll:
        bashContents = []
        bashContents.append("#!/bin/bash\n")
        bashContents.append("#SBATCH -n 12\n")
        bashContents.append("#SBATCH -N 1\n")
        bashContents.append("#SBATCH -p CLUSTER\n")
        bashContents.append("#SBATCH --mem=40GB\n")
        bashContents.append("#SBATCH -o outlag" + str(counter) + ".o\n")
        bashContents.append("#SBATCH -e outlag" + str(counter) + ".e\n")
        bashContents.append("cd /home/cywanag/liuxi/Documents/pyCode/paramagnet_interaction\n")
        bashContents.append(
            "python3 computeEigLag" + str(counter) + "part" + str(part)+"randseed"+str(rs) + ".py > part" + str(part) + "rec" + str(
                counter)+"Temp"+str(TVal) + ".txt\n")
        bsFileName = "lag" + str(counter) + ".sh"
        fbsTmp = open(bsFileName, "w+")
        for oneline in bashContents:
            fbsTmp.write(oneline)
        fbsTmp.close()
        counter+=1