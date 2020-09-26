import time
from datetime import datetime
from biosppy import ecg, eda
import numpy as np
from scipy import interpolate, signal, stats
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
from collections import OrderedDict
import sys

def countChecks(timestampFilePath, stageLabelList):
    timestampFile = open(timestampFilePath, "r")
    lines = timestampFile.readlines()
    timestampFile.close()
    if len(stageLabelList)*2 != len(lines):
        return 0
    return 1

def crop_stages(inFile, outFilePath, timestampList, stageLabelList):
    outFileList = []
    for stage in stageLabelList:
        tmpFile = open(''.join([outFilePath, stage, '.csv']), "w")
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

def extract_ecg_features(ecgFile, ecgFeatFile, freq, stage, winSize, shift):
    threshold = 0.2  # threshold for invalid window Ex) 0.2 represents 20%
    numOfFeatures = 8
    windowList = []
    window_begin = -1
    invalid_cnt = 0
    valid_cnt = 0
    firstTime = -1
    firstRead = 0
    while True:
        line = ecgFile.readline()  # 라인 by 라인으로 데이터 읽어오기
        if not line:
            # print(" End Of Zep ECG File")
            break
        lineData = [x.strip() for x in line.split(',')]  # 데이터 한 줄을 리스트로 변환
        currentTime = float(lineData[0])
        if firstRead == 0:
            firstTime = currentTime
            firstRead = 1

        if firstTime + shift <= currentTime:
            if window_begin == -1:
                window_begin = currentTime
            if lineData[1] != '':
                windowList.append(float(lineData[1]))
                valid_cnt += 1
            else:
                invalid_cnt += 1

            if currentTime - window_begin >= winSize:
                features = []
                if invalid_cnt <= ((winSize * freq) * threshold):  # valid window
                    try:
                        window = ecg.ecg(signal=windowList, sampling_rate=freq, show=False)
                    except ValueError:
                        for i in range(0, numOfFeatures):
                            features.append('n\t')
                    else:
                        # Calculate RR intervals
                        rrList = []
                        for i in range(1, len(window[2])):
                            rr = window[0][window[2][i]] - window[0][window[2][i - 1]]
                            rrList.append(rr)

                        # Calculate RMSSD that is heart rate variability (HRV)
                        hrvList = [0]
                        if rrList:
                            if len(rrList) > 1:
                                hrvList = []
                                for y in range(len(rrList) - 1):
                                    hrvList.append((rrList[y + 1] - rrList[y]) * (rrList[y + 1] - rrList[y]))
                        try:
                            freqDomFeat = frequencyDomain(rrList, band_type=None, lf_bw=0.11, hf_bw=0.1, plot=0)
                        except ValueError:
                            freqDomFeat = [0, 0, 0, 0]

                        # min-per20-med-per80-max-mean-std-hrv
                        # vlfe-lfe-hfe-lfhfr
                        features.append(str(np.min(rrList)) + "\t" + str(np.percentile(rrList, 20)) + "\t" + str(
                            np.median(rrList)) + "\t" + str(np.percentile(rrList, 80)) + "\t" + str(
                            np.max(rrList)) + "\t" + str(np.mean(rrList)) + "\t" + str(np.std(rrList)) + "\t" + str(
                            np.sqrt(np.average(hrvList))) + "\t")
                        # + "\t" + str(freqDomFeat[0]) + "\t" + str(freqDomFeat[1]) + "\t" + str(freqDomFeat[2]) + "\t" + str(freqDomFeat[3])
                else:  # equal to or more than 20% error? no features. that is invalid window
                    for i in range(0, numOfFeatures):
                        features.append('n\t')
                features.append(stage + '\n')
                ecgFeatFile.write(''.join(features))

                windowList = []
                window_begin = -1
                valid_cnt = 0
                invalid_cnt = 0


def frequencyDomain(RRints, band_type=None, lf_bw=0.11, hf_bw=0.1, plot=0):
    """ Computes frequency domain features on RR interval data

    Parameters:
    ------------
    RRints : list, shape = [n_samples,]
           RR interval data

    band_type : string, optional
             If band_type = None, the traditional frequency bands are used to compute
             spectral power:

                 LF: 0.003 - 0.04 Hz
                 HF: 0.04 - 0.15 Hz
                 VLF: 0.15 - 0.4 Hz

             If band_type is set to 'adapted', the bands are adjusted according to
             the protocol laid out in:

             Long, Xi, et al. "Spectral boundary adaptation on heart rate
             variability for sleep and wake classification." International
             Journal on Artificial Intelligence Tools 23.03 (2014): 1460002.

    lf_bw : float, optional
          Low frequency bandwidth centered around LF band peak frequency
          when band_type is set to 'adapted'. Defaults to 0.11

    hf_bw : float, optional
          High frequency bandwidth centered around HF band peak frequency
          when band_type is set to 'adapted'. Defaults to 0.1

    plot : int, 1|0
          Setting plot to 1 creates a matplotlib figure showing frequency
          versus spectral power with color shading to indicate the VLF, LF,
          and HF band bounds.

    Returns:
    ---------
    freqDomainFeats : dict
                   VLF_Power, LF_Power, HF_Power, LF/HF Ratio
    """

    # Remove ectopic beats
    # RR intervals differing by more than 20% from the one proceeding it are removed
    NNs = []
    for c, rr in enumerate(RRints):
        if abs(rr - RRints[c - 1]) <= 0.20 * RRints[c - 1]:
            NNs.append(rr)

    # Resample @ 4 Hz
    fsResamp = 4
    tmStamps = np.cumsum(NNs)  # in seconds
    f = interpolate.interp1d(tmStamps, NNs, 'cubic')
    tmInterp = np.arange(tmStamps[0], tmStamps[-1], 1 / fsResamp)
    RRinterp = f(tmInterp)

    # Remove DC component
    RRseries = RRinterp - np.mean(RRinterp)

    # Pwelch w/ zero pad
    fxx, pxx = signal.welch(RRseries, fsResamp, nfft=2 ** 14, window='hann')

    #original codes
    vlf = (0.003, 0.04)
    lf = (0.04, 0.15)
    hf = (0.15, 0.4)
    """
    vlf = (0.1, 0.2)
    lf = (0.2, 0.3)
    hf = (0.3, 0.4)
    """
    plot_labels = ['VLF', 'LF', 'HF']

    if band_type == 'adapted':

        vlf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])]))[0][0]]
        lf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])]))[0][0]]
        hf_peak = fxx[np.where(pxx == np.max(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])]))[0][0]]

        peak_freqs = (vlf_peak, lf_peak, hf_peak)

        hf = (peak_freqs[2] - hf_bw / 2, peak_freqs[2] + hf_bw / 2)
        lf = (peak_freqs[1] - lf_bw / 2, peak_freqs[1] + lf_bw / 2)
        vlf = (0.003, lf[0])

        if lf[0] < 0:
            print('***Warning***: Adapted LF band lower bound spills into negative frequency range')
            print('Lower thresold of LF band has been set to zero')
            print('Adjust LF and HF bandwidths accordingly')
            lf = (0, lf[1])
            vlf = (0, 0)
        elif hf[0] < 0:
            print('***Warning***: Adapted HF band lower bound spills into negative frequency range')
            print('Lower thresold of HF band has been set to zero')
            print('Adjust LF and HF bandwidths accordingly')
            hf = (0, hf[1])
            lf = (0, 0)
            vlf = (0, 0)

        plot_labels = ['Adapted_VLF', 'Adapted_LF', 'Adapted_HF']

    df = fxx[1] - fxx[0]
    vlf_power = np.trapz(pxx[np.logical_and(fxx >= vlf[0], fxx < vlf[1])], dx=df)
    lf_power = np.trapz(pxx[np.logical_and(fxx >= lf[0], fxx < lf[1])], dx=df)
    hf_power = np.trapz(pxx[np.logical_and(fxx >= hf[0], fxx < hf[1])], dx=df)
    totalPower = vlf_power + lf_power + hf_power

    # Normalize and take log
    vlf_NU_log = np.log((vlf_power / (totalPower - vlf_power)) + 1)
    lf_NU_log = np.log((lf_power / (totalPower - vlf_power)) + 1)
    hf_NU_log = np.log((hf_power / (totalPower - vlf_power)) + 1)
    lfhfRation_log = np.log((lf_power / hf_power) + 1)

    #freqDomainFeats = {'VLF_Power': vlf_NU_log, 'LF_Power': lf_NU_log,
    #                   'HF_Power': hf_NU_log, 'LF/HF': lfhfRation_log}
    freqDomainFeats = [vlf_NU_log, lf_NU_log, hf_NU_log, lfhfRation_log]

    if plot == 1:
        # Plot option
        freq_bands = {'vlf': vlf, 'lf': lf, 'hf': hf}
        freq_bands = OrderedDict(sorted(freq_bands.items(), key=lambda t: t[0]))
        colors = ['lightsalmon', 'lightsteelblue', 'darkseagreen']
        fig, ax = plt.subplots(1)
        ax.plot(fxx, pxx, c='grey')
        plt.xlim([0, 0.40])
        plt.xlabel(r'Frequency $(Hz)$')
        plt.ylabel(r'PSD $(s^2/Hz$)')

        for c, key in enumerate(freq_bands):
            ax.fill_between(
                fxx[min(np.where(fxx >= freq_bands[key][0])[0]): max(np.where(fxx <= freq_bands[key][1])[0])],
                pxx[min(np.where(fxx >= freq_bands[key][0])[0]): max(np.where(fxx <= freq_bands[key][1])[0])],
                0, facecolor=colors[c])

        patch1 = mpatches.Patch(color=colors[0], label=plot_labels[2])
        patch2 = mpatches.Patch(color=colors[1], label=plot_labels[1])
        patch3 = mpatches.Patch(color=colors[2], label=plot_labels[0])
        plt.legend(handles=[patch1, patch2, patch3])
        plt.show()

    return freqDomainFeats


def extract_gsr_features(gsrFile, gsrFeatFile, freq, stage, winSize, shift):
    onesets_threshold = 0.1
    numOfFeatures = 38
    EDA = [] # total
    stageEDA = [] # part corresponding to each stage
    beforeEDA = 0
    while True:
        line = gsrFile.readline()  # 라인 by 라인으로 데이터 읽어오기
        if not line:
            # print(" End Of Emp GSR File")
            break
        lineData = [x.strip() for x in line.split(',')]

        # triple redundancy because 4 Hz EDA data cannot be progressed by eda in biosppy (only works for more than or equal to 10 Hz)
        if lineData[1] != '' and float(lineData[1]) != 0:
            EDA.append(float(lineData[1]))
            EDA.append(float(lineData[1]))
            EDA.append(float(lineData[1]))
            beforeEDA = float(lineData[1])

            stageEDA.append(float(lineData[1]))
            stageEDA.append(float(lineData[1]))
            stageEDA.append(float(lineData[1]))
        else:
            EDA.append(beforeEDA)
            EDA.append(beforeEDA)
            EDA.append(beforeEDA)

            stageEDA.append(beforeEDA)
            stageEDA.append(beforeEDA)
            stageEDA.append(beforeEDA)

    allEDA = eda.eda(signal=EDA, sampling_rate=freq * 3, show=False)  # 3 represents triple redundancy
    if not stageEDA:
        slope = 0
    else:
        x = []
        for j in range(len(stageEDA)):
            x.append(j)
        s, intercept, r_value, p_value, std_err = stats.linregress(x, stageEDA)
        slope = s

    gsrFile.seek(0, 0)

    EDA_max = np.max(allEDA[1])
    EDA_min = np.min(allEDA[1])
    windowEDAList = []
    windowOnSetList = []
    windowPeakList = []
    windowAmplitudeList = []
    mappingList = []
    window_begin = -1
    cnt = 0
    onset = -1
    loop_count = 0
    last_level = 1
    firstRead = 0
    firstTime = -1
    while True:
        line = gsrFile.readline()  # 라인 by 라인으로 데이터 읽어오기
        if not line:
            # print(" End Of Emp GSR File")
            break
        lineData = [x.strip() for x in line.split(',')]
        currentTime = float(lineData[0])
        if firstRead == 0:
            firstTime = currentTime
            firstRead = 1

        if firstTime + shift <= currentTime:

            if window_begin == -1:
                window_begin = currentTime

            # need to examine 3 consecutive data for one timestamp since we duplicate raw data above 3 times
            for k in range(0,3):
                windowEDAList.append(allEDA[1][cnt + k])
                for i in range(0, len(allEDA[2])):  # examine if current EDA was determined as onset or not
                    if (cnt + k) == allEDA[2][i]:
                        onset = 1
                if onset == 1:  # if current EDA was determined as onset
                    if allEDA[1][
                        cnt + k] > onesets_threshold:  # AND if magnitude of current EDA is higher than threshold
                        onset_height = allEDA[1][cnt + k]
                        windowOnSetList.append(allEDA[1][cnt + k])  # store it in the onset list for current window
                        peak_height = np.max(allEDA[1][cnt + k:cnt + k + 50])
                        if peak_height > onset_height:  # Peak's height does not exist under Onset's height
                            windowPeakList.append(peak_height)  # store it in the peak list for current window
                            windowAmplitudeList.append(peak_height - onset_height)  # store it in the SCR amplitude list for current window
                        else:
                            windowPeakList.append(0)  # store it in the peak list for current window
                            windowAmplitudeList.append(
                                0)  # store it in the SCR amplitude list for current window
                onset = -1
            if currentTime - window_begin >= winSize:
                features = []
                loop_count += 1
                if windowEDAList.count(windowEDAList[0]) == winSize * 12:  # bad EDA period (constant line which is artificial)
                    for i in range(0, numOfFeatures):  # 38 dummy features
                        features.append('n\t')
                else:  # not bad EDA period
                    ###level Feature
                    Normal_windowEDAList = [Normalization(i, EDA_min, EDA_max) for i in windowEDAList]
                    ###subsampling in 5sec (60 sample)
                    subsampling = [np.mean(Normal_windowEDAList[i - 60:i]) for i in
                                   range(60, len(Normal_windowEDAList) + 1, 60)]
                    area = sum(subsampling)
                    # Dividing normalized EDA data into 5 states
                    for i in subsampling:
                        if i <= 0.2:
                            mappingList.append(1)
                        elif i <= 0.4:
                            mappingList.append(2)
                        elif i <= 0.6:
                            mappingList.append(3)
                        elif i <= 0.8:
                            mappingList.append(4)
                        else:
                            mappingList.append(5)
                    # global last_level
                    if loop_count == 1:
                        last_level = mappingList[0]
                    arou_num = unarou_num = 0
                    # number of arousing and unarousing moments and ratio between arousing, unarousing
                    for i in range(len(mappingList)):
                        if i == 0:
                            if mappingList[i] - last_level >= 1:
                                arou_num += 1
                            else:
                                unarou_num += 1
                        else:
                            if mappingList[i] - mappingList[i - 1] >= 1:
                                arou_num += 1
                            else:
                                unarou_num += 1
                    last_level = mappingList[len(mappingList) - 1]

                    # Total of 38 Features
                    # min-per20-med-per80-max-mean-std (EDA) 7
                    features.append(str(np.min(windowEDAList)) + "\t" + str(np.percentile(windowEDAList, 20)) + "\t" + str(np.median(windowEDAList)) + "\t" + str(np.percentile(windowEDAList, 80)) + "\t" \
                        + str(np.max(windowEDAList)) + "\t" + str(np.mean(windowEDAList)) + "\t" + str(np.std(windowEDAList)) + "\t")

                    if len(windowOnSetList) != 0:  # onSet appears!
                        # min-per20-med-per80-max-mean-std-num (onset) 8
                        # min-per20-med-per80-max-mean-std (peak) 7
                        # min-per20-med-per80-max-mean-std (amplitude) 7
                        features.append(str(np.min(windowOnSetList)) + "\t" + str(np.percentile(windowOnSetList, 20)) + "\t" + str(np.median(windowOnSetList)) + "\t" + str(np.percentile(windowOnSetList, 80)) + "\t" \
                                   + str(np.max(windowOnSetList)) + "\t" + str(np.mean(windowOnSetList)) + "\t" + str(np.std(windowOnSetList)) + "\t" + str(len(windowOnSetList)) + "\t" \
                                   + str(np.min(windowPeakList)) + "\t" + str(np.percentile(windowPeakList, 20)) + "\t" + str(np.median(windowPeakList)) + "\t" + str(np.percentile(windowPeakList, 80)) + "\t" \
                                   + str(np.max(windowPeakList)) + "\t" + str(np.mean(windowPeakList)) + "\t" + str(np.std(windowPeakList)) + "\t" \
                                   + str(np.min(windowAmplitudeList)) + "\t" + str(np.percentile(windowAmplitudeList, 20)) + "\t" + str(np.median(windowAmplitudeList)) + "\t" + str(np.percentile(windowAmplitudeList, 80)) + "\t" \
                                   + str(np.max(windowAmplitudeList)) + "\t" + str(np.mean(windowAmplitudeList)) + "\t" + str(np.std(windowAmplitudeList)) + "\t")
                    else:
                        for k in range(8+7+7):
                            features.append(str(0) + "\t")

                    # level_i (1,2,3,4,5) -> Ratio between the number of level_i and for each 30 second window 5
                    # mean of level, area of subsampling 2
                    features.append(str(mappingList.count(1) / len(mappingList)) + "\t" + str(mappingList.count(2) / len(mappingList)) + "\t" \
                                   + str(mappingList.count(3) / len(mappingList)) + "\t" + str(mappingList.count(4) / len(mappingList)) + "\t" \
                                   + str(mappingList.count(5) / len(mappingList)) + "\t" + str(np.mean(mappingList)) + "\t" + str(area) + "\t")

                    # sign bit of slope (EDA) & slope (EDA)
                    if slope > 0:
                        features.append("1\t" + str(slope) + "\t")
                    elif slope < 0:
                        features.append("-1\t" + str(slope) + "\t")
                    else:
                        features.append("0\t" + str(slope) + "\t")
                # label
                features.append(stage + "\n")
                gsrFeatFile.write(''.join(features))

                window_begin = -1
                windowEDAList = []
                windowOnSetList = []
                windowPeakList = []
                windowAmplitudeList = []
                Normal_windowEDAList = []
                mappingList = []
            cnt += 3

def Normalization(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)

def extract_resp_features(respFile, respFeatFile, freq, stage, winSize, shift):
    threshold = 0.2  # threshold for invalid window Ex) 0.2 represents 20%
    numOfFeatures = 63
    firstTime = -1
    firstRead = 0
    windowList = []
    window_begin = -1
    invalid_cnt = 0
    valid_cnt = 0

    while True:
        line = respFile.readline()  # 라인 by 라인으로 데이터 읽어오기
        if not line:
            # print(" End Of Zep Resp. File")
            break
        lineData = [x.strip() for x in line.split(',')]  # 데이터 한 줄을 리스트로 변환
        currentTime = float(lineData[0])
        if firstRead == 0:
            firstTime = currentTime
            firstRead = 1

        if firstTime + shift <= currentTime:
            if window_begin == -1:
                window_begin = currentTime
            if lineData[1] != '':
                windowList.append(float(lineData[1]))
                valid_cnt += 1
            else:
                invalid_cnt += 1

            if currentTime - window_begin >= winSize:
                features = []

                waves = []
                waves2 = []
                peak_x = []
                peak_y = []
                valley_y = []
                valley_x = []
                up_intercept = []
                up_intercepty = []
                down_intercept = []
                down_intercepty = []
                flag = []
                dupli = []
                total_intercept = []

                insp_list = []
                exp_list = []
                ieratio_list = []
                total_list = []
                br_list = []
                stretch = []
                MI = []
                ME = []
                MD = []
                stretch2 = []

                if invalid_cnt <= ((winSize * freq) * threshold):  # valid window
                    try:
                        window = smoothing(waves, data=windowList, size=6, step=1)  # Signal Smoothing
                        # window = list(window)
                    except IndexError:  # manage IndexError occurring inside of resp.resp function
                        for i in range(0, numOfFeatures):
                            features.append("n\t")
                        idx = -1
                    else:  # no IndexError inside of resp.resp function

                        # Moving Average Centerline(MAC)
                        # window size 가 30보다 작으면 moving average 의 단위보다 작아져서 안 됨 - 200926 Moon 발견
                        #series = MAC(waves2, data=window, size=30, step=1)
                        series = MAC(waves2, data=window, size=winSize, step=1) # 200926

                        # Intercept Identification
                        for i in range(len(series) - 1):
                            if window[i] <= series[i + 1] <= window[i + 1]:
                                up_intercept.append(i)
                                flag.append(0)
                                up_intercepty.append(series[i])
                                total_intercept.append((i, series[i]))
                            elif window[i + 1] <= series[i + 1] <= window[i]:
                                down_intercept.append(i)
                                flag.append(1)
                                down_intercepty.append(series[i])
                                total_intercept.append((i, series[i]))

                        # Intercept Screening
                        for i in range(len(flag) - 1):
                            if flag[i] == flag[i + 1]:
                                dupli.append(total_intercept[i])
                        for i in dupli:
                            total_intercept.remove(i)
                        total_intercept = flatten(total_intercept)
                        intercept = total_intercept[::2]
                        # intercepty = total_intercept[1::2]

                        for i in range(len(intercept) - 1):

                            # Peak Detection
                            if window[intercept[i]] < window[intercept[i] + 1]:
                                a = []
                                for j in range(intercept[i], intercept[i + 1]):
                                    a.append((j, window[j]))
                                a = flatten(a)
                                c = np.max(a)
                                peak_y.append(c)
                                peak_x.append(a[a.index(c) - 1])
                            # Valley Detection
                            else:
                                a = []
                                for j in range(intercept[i], intercept[i + 1]):
                                    a.append((j, window[j]))
                                a = flatten(a)
                                b = a[1::2]
                                c = np.min(b)
                                valley_y.append(c)
                                valley_x.append(a[a.index(c) - 1])

                        if len(peak_x) < len(valley_x):
                            deci = peak_x
                        else:
                            deci = valley_x
                        if len(peak_x) == 0 or len(valley_x) == 0:
                            for i in range(0, numOfFeatures):
                                features.append("n\t")
                        elif peak_x[0] < valley_x[0]:
                            for x in range(len(deci) - 1):
                                insp = peak_x[x + 1] - valley_x[x]
                                insp_list.append(insp)
                                exp = valley_x[x + 1] - peak_x[x + 1]
                                exp_list.append(exp)

                                MI.append(peak_y[x + 1] - valley_y[x])
                                ME.append(peak_y[x + 1] - valley_y[x + 1])
                                MD.append(abs((peak_y[x + 1] - valley_y[x]) - (peak_y[x + 1] - valley_y[x + 1])))
                                if exp == 0:
                                    ieratio_list.append(0)
                                else:
                                    ieratio_list.append(insp / exp)
                                    total_list.append(insp + exp)
                                if valley_y[x + 1] < valley_y[x]:
                                    stretch.append(peak_y[x + 1] - valley_y[x + 1])
                                else:
                                    stretch.append(peak_y[x + 1] - valley_y[x])
                                avg_stretch = np.mean(stretch)

                                if valley_y[x + 1] < valley_y[x]:
                                    if avg_stretch * 0.2 < (peak_y[x + 1] - valley_y[x + 1]):
                                        stretch2.append(peak_y[x + 1] - valley_y[x + 1])
                                else:
                                    if avg_stretch * 0.2 < (peak_y[x + 1] - valley_y[x]):
                                        stretch2.append(peak_y[x + 1] - valley_y[x])
                            br_list.append(len(insp_list))

                        else:
                            for x in range(len(deci) - 1):
                                insp = peak_x[x] - valley_x[x]
                                insp_list.append(insp)
                                exp = valley_x[x + 1] - peak_x[x]
                                exp_list.append(exp)

                                MI.append(peak_y[x] - valley_y[x])
                                ME.append(peak_y[x] - valley_y[x + 1])
                                MD.append(abs((peak_y[x] - valley_y[x]) - (peak_y[x] - valley_y[x + 1])))
                                if exp == 0:
                                    ieratio_list.append(0)
                                else:
                                    ieratio_list.append(insp / exp)
                                    total_list.append(insp + exp)

                                if valley_y[x + 1] < valley_y[x]:
                                    stretch.append(peak_y[x] - valley_y[x + 1])
                                else:
                                    stretch.append(peak_y[x] - valley_y[x])
                                avg_stretch = np.mean(stretch)

                                if valley_y[x + 1] < valley_y[x]:
                                    if avg_stretch * 0.2 < (peak_y[x] - valley_y[x + 1]):
                                        stretch2.append(peak_y[x] - valley_y[x + 1])
                                else:
                                    if avg_stretch * 0.2 < (peak_y[x] - valley_y[x]):
                                        stretch2.append(peak_y[x] - valley_y[x])
                            br_list.append(len(insp_list))

                        if len(insp_list) == 0 or len(exp_list) == 0:
                            for i in range(0, numOfFeatures):
                                features.append("n\t")
                        else:
                            # min-per20-med-per80-max-mean-std (duration of inspiration)
                            # min-per20-med-per80-max-mean-std (duration of expiration)
                            # min-per20-med-per80-max-mean-std (I:E duration ratio)
                            # min-per20-med-per80-max-mean-std (duration of respiration)
                            # min-per20-med-per80-max-mean-std (breathing rate)
                            # min-per20-med-per80-max-mean-std (Inspiration Magnitude)
                            # min-per20-med-per80-max-mean-std (Expiration Magnitude)
                            # min-per20-med-per80-max-mean-std (Magnitude Difference(|MI-MD|))
                            # min-per20-med-per80-max-mean-std (stretch)
                            features.append(str(np.min(insp_list)) + "\t" + str(
                                np.percentile(insp_list, 20)) + "\t" + str(np.median(insp_list)) + "\t" + str(
                                np.percentile(insp_list, 80)) + "\t" + str(np.max(insp_list)) + "\t" + str(
                                np.mean(insp_list)) + "\t" + str(np.std(insp_list)) + "\t" + str(
                                np.min(exp_list)) + "\t" + str(np.percentile(exp_list, 20)) + "\t" + str(
                                np.median(exp_list)) + "\t" + str(np.percentile(exp_list, 80)) + "\t" + str(
                                np.max(exp_list)) + "\t" + str(np.mean(exp_list)) + "\t" + str(
                                np.std(exp_list)) + "\t" + str(
                                np.min(ieratio_list)) + "\t" + str(np.percentile(ieratio_list, 20)) + "\t" + str(
                                np.median(ieratio_list)) + "\t" + str(np.percentile(ieratio_list, 80)) + "\t" + str(
                                np.max(ieratio_list)) + "\t" + str(np.mean(ieratio_list)) + "\t" + str(
                                np.std(ieratio_list)) + "\t" + str(
                                np.min(total_list)) + "\t" + str(np.percentile(total_list, 20)) + "\t" + str(
                                np.median(total_list)) + "\t" + str(np.percentile(total_list, 80)) + "\t" + str(
                                np.max(total_list)) + "\t" + str(np.mean(total_list)) + "\t" + str(
                                np.std(total_list)) + "\t" + str(
                                np.min(br_list)) + "\t" + str(np.percentile(br_list, 20)) + "\t" + str(
                                np.median(br_list)) + "\t" + str(np.percentile(br_list, 80)) + "\t" + str(
                                np.max(br_list)) + "\t" + str(np.mean(br_list)) + "\t" + str(
                                np.std(br_list)) + "\t" + str(
                                np.min(MI)) + "\t" + str(np.percentile(MI, 20)) + "\t" + str(
                                np.median(MI)) + "\t" + str(np.percentile(MI, 80)) + "\t" + str(
                                np.max(MI)) + "\t" + str(np.mean(MI)) + "\t" + str(np.std(MI)) + "\t" + str(
                                np.min(ME)) + "\t" + str(np.percentile(ME, 20)) + "\t" + str(
                                np.median(ME)) + "\t" + str(np.percentile(ME, 80)) + "\t" + str(
                                np.max(ME)) + "\t" + str(np.mean(ME)) + "\t" + str(np.std(ME)) + "\t" + str(
                                np.min(MD)) + "\t" + str(np.percentile(MD, 20)) + "\t" + str(
                                np.median(MD)) + "\t" + str(np.percentile(MD, 80)) + "\t" + str(
                                np.max(MD)) + "\t" + str(np.mean(MD)) + "\t" + str(np.std(MD)) + "\t" + str(
                                np.min(stretch2)) + "\t" + str(np.percentile(stretch2, 20)) + "\t" + str(
                                np.median(stretch2)) + "\t" + str(np.percentile(stretch2, 80)) + "\t" + str(
                                np.max(stretch2)) + "\t" + str(np.mean(stretch2)) + "\t" + str(np.std(stretch2)) + "\t")
                elif invalid_cnt > ((winSize * freq) * threshold):  # equal to or more than 20% error? no features. that is invalid window
                    for i in range(0, numOfFeatures):
                        features.append("n\t")
                features.append(stage + "\n")
                respFeatFile.write(''.join(features))
                windowList = []
                invalid_cnt = 0
                window_begin = -1
                valid_cnt = 0
        else:  # NOT btw each phase of the entire experiment
            window_begin = -1
            windowList = []

def peakdet(v, delta, x=None):
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lookformax = True
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return np.array(maxtab), np.array(mintab)

def flatten(x):
    basestring = (str, bytes)
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            # 스트링형이 아닌 이터러블의 경우 - 재귀한 후, extend
            result.extend(flatten(el))  # 재귀 호출
        else:
            # 이터러블이 아니거나 스트링인 경우 - 그냥 append
            result.append(el)
    return result

def smoothing(waves, data, size, step):
    # check input

    if data is None:
        raise TypeError("Please specify an input data set.")
    if size is None:
        raise TypeError("Please specify the number of samples for the median.")
    if step is None:
        step = size
    if step < 0:
        raise ValueError("The step must be a positive integer.")
        # compute
    waves = [np.mean(data[i - size:i + size + 1]) for i in range(0, len(data))]
    waves = [x for x in waves if str(x) != 'nan']
    return waves

def MAC(waves2, data, size, step):
    # check inputs
    if data is None:
        raise TypeError("Please specify an input data set.")
    if size is None:
        raise TypeError("Please specify the number of samples for the median.")
    if step is None:
        step = size
    if step < 0:
        raise ValueError("The step must be a positive integer.")
        # number of waves
    L = len(data) - size
    nb = 1 + L // step
    if nb <= 0:
        raise ValueError("Not enough samples for the given `size`.")
        # compute
    for i in range(0, size):
        waves2.append(np.mean(data[0:2 * size + 1]))
    for i in range(size, L + 1):
        waves2.append(np.mean(data[i - size:i + size + 1]))
    for i in range(L + 1, len(data)):
        waves2.append(np.mean(data[len(data) - 2 * size:len(data) + 1]))
    return waves2

