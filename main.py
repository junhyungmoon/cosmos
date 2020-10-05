#from numpy import *
import numpy as np
import scipy.stats
from codes.functions import countChecks, crop_stages, extract_ecg_features, extract_gsr_features, extract_resp_features

# Flags
examineFlag = 0
cropFlag = 0
extractFlag = 0
parseFlag = 0
analyzeFlag = 0
correlFlag = 1

### Global variables ###
PcortisolScoreList = [3,4,6,7,9,10,11,12,13,14,16,17,18,20,21,23,24,26,28,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,63,64,65,100]# Indices of 40 subjects

femaleList = [3,14,17,24,31,33,36,37,38,39,40,41,43,44,45,46,63] # 17
maleList = [4,6,7,9,10,11,12,13,16,18,20,21,23,26,28,30,32,34,35,42,64,65,100] # 23
twentyList = [3,4,6,7,9,10,11,12,13,14,18,20,21,23,24,26,28,30,31,32,34,35,36,38,39,40,42,43,44,45,64,65,100] # 33
thirtyList = [16,17,33,37,41,46,63] # 7
randList = [5,9,10,13,16,23,24,25,26,28,29,30,31,32,33,34,35,36,38,39] #20
# Feature extraction error: 17(ck), 20(s2), 65('vs', 'e1', 'e2', 'r2') in E_gsr
# 17 - no reason
# 20 - invalid data for 's2' in E_gsr
# 65 - invalid data for 'vs', 'e1', 'e2', 'r2' in E_gsr
postfixList = ['Z_bre','Z_ecg']#'E_gsr']#,
#rootPath = "E:\mjh\study\cosmos\data\\2nd_exp\\"
rootPath = "E:\mjh\study\\bibm\\features\\"
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


# 4. Examine important features according to each experimental stage for each individual
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

if analyzeFlag:
    try:
        srcFile = open(''.join([sensorFilePath, 'idx_age_sex_scores_cortisols_edamean_inspmin_rrmin.txt']), "r")
        # idx_age_sex_scores(9)_cortisols(3)_edamean(9)_inspmin(9)_rrmin(9)
    except FileNotFoundError as e:
        print("No file error")
    else:
        line = srcFile.readline()
        lineData = [x.strip() for x in line.split('\t')]
        srcFile.seek(0,0)
        dataBuf = []
        cntBuf = []
        for k in range(len(lineData)-3): #exclude idx, age, and sex
            dataBuf.append(0)
            cntBuf.append(0)

        while True:
            line = srcFile.readline()
            if not line:
                break
            lineData = [x.strip() for x in line.split('\t')]
            if int(lineData[0]) in randList:
                for k in range(3, len(lineData)):
                    if lineData[k] != 'n':
                        dataBuf[k-3] += float(lineData[k])
                        cntBuf[k-3] += 1

        for k in range(len(lineData)-3): #exclude idx, age, and sex
            dataBuf[k] = str(dataBuf[k]/cntBuf[k]) + "\t"
        print(''.join(dataBuf))
        #for a in range(20):
            #print(random.randint(1,41))
        srcFile.close()

if correlFlag:
    try:
        srcFile = open(''.join([sensorFilePath, 'idx_age_sex_scores_cortisols_edamean_inspmin_rrmin.txt']), "r")
        # idx_age_sex_scores(9)_cortisols(3)_edamean(9)_inspmin(9)_rrmin(9)
    except FileNotFoundError as e:
        print("No file error")
    else:
        line = srcFile.readline()
        lineData = [x.strip() for x in line.split('\t')]
        srcFile.seek(0,0)

        allBuf = []
        femaleBuf = []
        maleBuf = []
        twentyBuf = []
        thirtyBuf = []
        randBuf = []

        while True:
            line = srcFile.readline()
            if not line:
                break
            lineData = [x.strip() for x in line.split('\t')]
            # 20, 36, 65 invalid values in several stages (feature - edamean, inspmin, RRmin)
            if int(lineData[0]) not in [20, 36, 65]:
                if int(lineData[0]) in PcortisolScoreList:
                    allBuf.append([float(i) for i in lineData[3:]]) # idx, age, sex
                if int(lineData[0]) in femaleList:
                    femaleBuf.append([float(i) for i in lineData[3:]]) # idx, age, sex
                if int(lineData[0]) in maleList:
                    maleBuf.append([float(i) for i in lineData[3:]])
                if int(lineData[0]) in twentyList:
                    twentyBuf.append([float(i) for i in lineData[3:]])
                if int(lineData[0]) in thirtyList:
                    thirtyBuf.append([float(i) for i in lineData[3:]])
                if int(lineData[0]) in randList:
                    randBuf.append([float(i) for i in lineData[3:]])
        srcFile.close()

        srcList = [allBuf, femaleBuf, maleBuf, twentyBuf, thirtyBuf, randBuf]
        destList = [allBuf, femaleBuf, maleBuf, twentyBuf, thirtyBuf, randBuf]

        for src in srcList:
            avgCoeff = 0
            cnt = 0
            for a in range(len(src)):
                for b in range(len(src)):
                    if a == b:
                        continue
                    avgCoeff += scipy.stats.pearsonr(np.array(src[a]), np.transpose(np.array(src[b])))[0]
                    cnt += 1
            print(avgCoeff/cnt)

        for src in srcList:
            for dest in destList:
                if src == dest:
                    continue
                avgCoeff = 0
                cnt = 0
                for a in range(len(src)):
                    for b in range(len(dest)):
                        avgCoeff += scipy.stats.pearsonr(np.array(src[a]), np.transpose(np.array(dest[b])))[0]
                        cnt += 1
                print(avgCoeff/cnt)