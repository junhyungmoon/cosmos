from numpy import *
from codes.functions import countChecks, crop_stages, extract_ecg_features, extract_gsr_features, extract_resp_features

# Flags
examineFlag = 0
cropFlag = 0
extractFlag = 0
parseFlag = 1

### Global variables ###
PcortisolScoreList = [3,4,6,7,9,10,11,12,13,14,16,17,18,20,21,23,24,26,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,63,64,65,100]# Indices of 40 subjects
# Feature extraction error: 17(ck), 20(s2), 65('vs', 'e1', 'e2', 'r2') in E_gsr
# 17 - no reason
# 20 - invalid data for 's2' in E_gsr
# 65 - invalid data for 'vs', 'e1', 'e2', 'r2' in E_gsr
postfixList = ['Z_bre','Z_ecg']#'E_gsr']#,
#rootPath = "E:\mjh\study\cosmos\data\\2nd_exp\\"
rootPath = "E:\mjh\study\\bibm\\features\\15\\"
sensorFilePath = rootPath
timeFilePath = rootPath + "timestamp\\"
stageLabelList = ['r1', 'ck', 's1', 'ip', 'ie', 's2', 'vs', 'e1', 'e2', 'r2'] #

# 1. examination result: P20 and P65 misses several timestamps of CHECKs (200918)
if examineFlag:
    # add dummy timestamps for missing periods (200918)
    for subNum in PcortisolScoreList:
        result = countChecks(''.join([timeFilePath, 'P', str(subNum), '_t.txt']), stageLabelList)
        if result != 1:
            print(''.join(['Timestamps of several stages are missing: Subject ', str(subNum)]))


# 2. Crop partial data according to the beginning and the end of each experimental stage
if cropFlag:
    for subNum in PcortisolScoreList:
        try:
            timeFile = open(''.join([timeFilePath, 'P', str(subNum), '_t.txt']), "r")
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

            # Parse data of GSR(from Empatica E4), ECG(from Zephyr Bioharness), and RESP(from Zephyr Bioharness)
            for postfix in postfixList:
                sensorFileName = "P" + str(subNum) + "_" + postfix
                sensorFile = open(sensorFilePath + sensorFileName + ".csv", "r")
                outFilePath = rootPath + "Cropped files\\" + sensorFileName + "_"
                crop_stages(sensorFile, outFilePath, timestampList, stageLabelList)
                sensorFile.close()



# 3. Extract features from cropped files
if extractFlag:
    for subNum in PcortisolScoreList:
        for postfix in postfixList:
            for stage in stageLabelList:
                try:
                    srcFile = open(''.join([sensorFilePath, 'Cropped files\\P', str(subNum), '_', postfix, '_', stage, '.csv']), "r")
                    destFile = open(''.join([sensorFilePath, 'Features\\P', str(subNum), '_', postfix, '_', stage, '.csv']), "w")
                except FileNotFoundError as e:
                    print("No file error: Subject " + str(subNum))
                else:
                    winSize = 30
                    shift = 0
                    if postfix == 'Z_ecg':
                        freq = 250  # 4 # 250
                        extract_ecg_features(srcFile, destFile, freq, stage, winSize, shift)
                    if postfix == 'E_gsr':
                        if subNum == 17 and stage == 'ck':
                            continue
                        if subNum == 20 and stage == 's2':
                            continue
                        if subNum == 65 and stage in ['vs', 'e1', 'e2', 'r2']:
                            continue
                        freq = 4
                        extract_gsr_features(srcFile, destFile, freq, stage, winSize, shift)
                    if postfix == 'Z_bre':
                        freq = 250  # 4 # 250
                        extract_resp_features(srcFile, destFile, freq, stage, winSize, shift)
                    srcFile.close()
                    destFile.close()


# 4. Examine features according to each experimental stage
if parseFlag:
    # the most important features in each sensor
    # edamean (6번째)
    # inspmin (1번째)
    # RRmin (1번째)
    # index: 0 (RRmin), 8 (inspmin), 76 (edamean)
    for postfix in postfixList:
        for subNum in PcortisolScoreList:
            featureBuf = []
            for stage in stageLabelList:
                try:
                    srcFile = open(''.join([sensorFilePath, 'P', str(subNum), '_test.arff']), "r")
                except FileNotFoundError as e:
                    print("No file error: Subject " + str(subNum))
                else:
                    sum = 0 # RRmin
                    cnt = 0
                    while True:
                        line = srcFile.readline()
                        if not line:
                            break
                        lineData = [x.strip() for x in line.split(',')]
                        if lineData[-1] == stage:
                            if postfix == 'Z_ecg':
                                sum += float(lineData[0])
                            if postfix == 'Z_bre':
                                sum += float(lineData[8])
                            if postfix == 'E_gsr':
                                sum += float(lineData[76])
                            cnt += 1
                    if cnt == 0:
                        featureBuf.append("n\t")
                    else:
                        featureBuf.append(str(sum/cnt)+"\t")
                    srcFile.close()
            print(''.join(featureBuf))
