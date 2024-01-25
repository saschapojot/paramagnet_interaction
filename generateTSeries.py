import numpy as np
import re

#this script generates MCMC computing scripts as well as bash files for submitting jobs


lagFileName="computeEigLag"
suffix=".py"

part=5
TemperaturesAll=[0.1+0.05*n  for n in range(0,51)]
randSeedAll=[433]


fileIn=open(lagFileName+suffix,"r")

contents=fileIn.readlines()
lineTemperature=0#the line corresponding to T=xxxxx (temperature)
lineRandSeed=0# random seed
lineMaxStep=0# loop numbers in first mc
linePart=0
for l in range(0,len(contents)):
    line=contents[l]
    if re.findall("^T=\d+",line):
        lineTemperature=l
        # print(lineTemperature)
    if re.findall("^random\.seed",line):
        lineRandSeed=l
        # print(lineRandSeed)
    if re.findall("^maxEquilbrationStep=\d+",line):
        lineMaxStep=l
    if re.findall("^part=\d+",line):
        linePart=l


        # print(lineMaxEq)





# contents[-3]='outPklFileName="T"+str(T)+"t"+str(t)+"J"+str(J)+"g"+str(g)+"part"+str('+str(part)+')+"out.pkl"\n'
#
#
# maxStepMatch=re.search("\d+",contents[lineMaxStep])
# if maxStepMatch:
#     maxStep=maxStepMatch.group()
setMaxStep=50000
counter=0
for TVal in TemperaturesAll:
    for rs in randSeedAll:
        contents[lineTemperature]="T="+str(TVal)+"\n"
        contents[lineRandSeed]="random.seed("+str(rs)+")\n"
        contents[linePart]="part="+str(part)+"\n"
        contents[lineMaxStep]="maxEquilbrationStep="+str(setMaxStep)+"\n"
        contents[-5]='outDir="./part"+str(part)+"/"\n'
        contents[-4]='Path(outDir).mkdir(parents=True, exist_ok=True)\n'
        contents[-3] = 'outPklFileName=outDir+"T"+str(T)+"t"+str(t)+"J"+str(J)+"g"+str(g)'+'+"randseed"+str('+str(rs)+')+"step"+str('+str(setMaxStep)+')+"part"+str(' + str(
            part) + ')+"out.pkl"\n'
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
        bashContents.append("#SBATCH -t 0-40:00\n")
        bashContents.append("#SBATCH -p CLUSTER\n")
        bashContents.append("#SBATCH --mem=40GB\n")
        bashContents.append("#SBATCH -o outlag" + str(counter) + ".o\n")
        bashContents.append("#SBATCH -e outlag" + str(counter) + ".e\n")
        bashContents.append("cd /home/cywanag/liuxi/Documents/pyCode/paramagnet_interaction\n")
        bashContents.append(
            "python3 computeEigLag" + str(counter) + "part" + str(part)+"randseed"+str(rs) + ".py > part" + str(part) + "rec" + str(
                counter)+"Temp"+str(TVal) + ".txt\n")
        bsFileName = "./lagBash/lag" + str(counter) + ".sh"
        fbsTmp = open(bsFileName, "w+")
        for oneline in bashContents:
            fbsTmp.write(oneline)
        fbsTmp.close()
        counter+=1