from numpy import *
from codes.functions import crop_stages, extract_ecg_features, extract_gsr_features, extract_resp_features

PcortisolScoreList = [3]#,4,6,7,9,10,11,12,13,14,16,17,18,20,21,23,24,26,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,63,64,65,100] # Indices of 40 subjects
postfixList = ['E_gsr']#, 'Z_ecg', 'Z_bre']

# 1. Crop partial data according to the beginning and the end of each experimental stage
rootPath = "E:\mjh\study\cosmos\data\\2nd_exp\\"
sensorFilePath = rootPath
timeFilePath = rootPath + "timestamp\\"
for subNum in PcortisolScoreList:
    try:
        timeFileName = "P" + str(subNum) + "_t.txt"
        timeFile = open(timeFilePath + timeFileName, "r")
    except FileNotFoundError as e:
        print("No file error: Subject " + str(subNum))
    else:
        # read timestamps of "CHECK"s which are either the beginning or the end of each experimental stage
        timestampList = []
        while True:
            line = timeFile.readline()
            if not line:
                break
            lineData = [x.strip() for x in line.split('\n')]
            timestampList.append(float(lineData[0]))
        timeFile.close()
        numOfStages = int(len(timestampList) / 2)  # number of experimental stages

        for postfix in postfixList:
            sensorFileName = "P" + str(subNum) + "_" + postfix
            sensorFile = open(sensorFilePath + sensorFileName + ".csv", "r")

            stageLabelList = ['r1', 'ck', 's1', 'ip', 'ie', 's2', 'vs', 'e1', 'e2', 'r2']
            outFilePath = rootPath + "Cropped files\\" + sensorFileName + "_"
            crop_stages(sensorFile, outFilePath, timestampList, stageLabelList)

            sensorFile.close()


