import time
from datetime import datetime
from biosppy import ecg
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.patches as mpatches
from collections import OrderedDict

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
                        rrList = []
                        for i in range(1, len(window[2])):
                            rr = window[0][window[2][i]] - window[0][window[2][i - 1]]
                            rrList.append(rr)

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

def extract_gsr_features(gsrFile, gsrFeatFile, winSize, shift):
    ### Store timestamps of CHECKs to buffer
    timearray = []

def extract_resp_features(respFile, respFeatFile, winSize, shift):
    ### Store timestamps of CHECKs to buffer
    timearray = []

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