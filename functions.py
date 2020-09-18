def crop_stages(inFile, outFilePath, timestampList, stageLabelList):
    outFileList = []
    for stage in stageLabelList:
        tmpFile = open(outFilePath + stage + ".csv", "w")
        outFileList.append(tmpFile)

    stageIdx = -1
    while True:
        line = inFile.readline()
        if not line:
            break
        lineData = [x.strip() for x in line.split(',')]
        for a in range(len(stageLabelList)):
            if timestampList[a*2] <= float(lineData[0]) <= timestampList[a*2+1]:
                stageIdx = a
                break
            else:
                stageIdx = -1
        if stageIdx >= 0:
            outFileList[stageIdx].write(line)

    for fileIdx in range(len(stageLabelList)):
        outFileList[fileIdx].close()

def extract_ecg_features(inFile, outFile, winSize, shift):
    ### Store timestamps of CHECKs to buffer
    timearray = []

def extract_gsr_features(inFile, outFile, winSize, shift):
    ### Store timestamps of CHECKs to buffer
    timearray = []

def extract_resp_features(iinFile, outFile, winSize, shift):
    ### Store timestamps of CHECKs to buffer
    timearray = []