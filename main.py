from numpy import *
from codes.functions import crop_stages, extract_ecg_features, extract_gsr_features, extract_resp_features


PcortisolScoreList = [3,4,6,7,9,10,11,12,13,14,16,17,18,20,21,23,24,26,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,63,64,65,100] # 40 subjects

crop_stages()

# 1. Crop partial data according to the beginning and the end of each experimental stage
rootPath = "D:\strEsstimate\exp_data_2nd\\"
sensorFilePath = rootPath + "D:\strEsstimate\exp_data_2nd\\"
timeFilePath = rootPath + "D:\strEsstimate\exp_data_2nd\\"
try:
    sensorFile = open(sensorFilePath, "r")
    timeFile = open(timeFilePath, "r")
except FileNotFoundError as e:
    print("No file error: " + sensorFilePath)
else:
    # read timestamps of "CHECK"s which are either the beginning or the end of each experimental stage
    timestampArr = []
    while True:
        line = timeFile.readline()
        if not line:
            break

        lineData = [x.strip() for x in line.split('\n')]
        timestampArr.append(float(lineData[0]))
    timeFile.close()
    numOfStages = int(len(timestampArr) / 2)  # number of experimental stages

    stageLabelArr = ['r1', 'ck', 's1', 'ip', 'ie', 's2', 'vs', 'e1', 'e2', 'r2']

    firstIdx = 0
    lastIdx = numOfStages
    if int(len(stageLabelArr)) != numOfStages:
        firstIdx = 1
        lastIdx = numOfStages

    for stage in stageLabelArr:
        outFile = open(rootPath + stage, "w")
        crop_stages(sensorFile, outFile)
        sensorFile.seek(0,0)
        outFile.close()


