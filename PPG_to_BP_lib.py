# -----------------------------------------------
# PPG-BP signal functions
# 
# This library contains several functions to interface with the
# pyPPG library for processing PPG-BP signals. This includes loading,
# preprocessing, fiducial point detection, and biomarker extraction.
#
# These functions are needed for short PPG segments, e.g., single beats.
#
# Additionally, a modified version of the QPPG beat detection algorithm
# is included to detect beats in short PPG segments.
# -----------------------------------------------

import pandas as pd
import numpy as np
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
from pyPPG.ppg_bm.bm_extraction import BmExctator
from dotmap import DotMap


# -----------------------------------------------
# Load PPG-BP signal from text file
# -----------------------------------------------
def load_signal_PPGBP(file_id, data_dir, fs):
    
    data_path = data_dir + file_id + '.txt'
    data = pd.read_csv(data_path,delimiter='\t',header=None).to_numpy().squeeze()
    time_sec = np.arange(len(data))/fs

    return data, time_sec

# -----------------------------------------------
# Preprocess PPG-BP signal
# -----------------------------------------------
def preprocess_ppgbp(data,fs,file_id, fL, fH, order, sm_wins):
    fs_data = fs
    start_sig = 0 # the first sample of the signal to be analysed
    end_sig = -1 # the last sample of the signal to be analysed (here a value of '-1' indicates the last sample)

    s = DotMap()
    s.fs = fs
    s.name = file_id
    s.v = data[:-1]
    s.ppg = data[:-1]
    s.start_sig = start_sig
    s.end_sig = end_sig

    #preprocess signal
    s.filtering = True # whether or not to filter the PPG signal
    s.fL=fL # Lower cutoff frequency (Hz)
    s.fH=fH # Upper cutoff frequency (Hz)
    s.order=order # Filter order
    s.sm_wins = sm_wins

    prep = PP.Preprocess(fL=s.fL, fH=s.fH, order=s.order, sm_wins=s.sm_wins)
    s.ppg, s.vpg, s.apg, s.jpg = prep.get_signals(s=s)

    return s

# -----------------------------------------------
# Merge fiducial points into a single DataFrame
# -----------------------------------------------
def merge_fiducials(ppg_fp, vpg_fp, apg_fp, jpg_fp):
    fiducials = pd.DataFrame()
    for temp_sig in (ppg_fp, vpg_fp, apg_fp, jpg_fp):
        for key in list(temp_sig.keys()):
            fiducials[key] = temp_sig[key].values

    return fiducials

# -----------------------------------------------
# Find fiducial points in a PPG beat
# -----------------------------------------------
def find_fiducials(s,onsets):
    s.correct = True
    s_class = PPG(s, check_ppg_len=False)

    ## Create a fiducial class
    fpex = FP.FpCollection(s_class)

    # Extract fiducial points
    peak = [np.argmax(s.ppg[onsets[0]:onsets[1]]) + onsets[0]]

    ppg_fp = pd.DataFrame()
    ppg_fp['dn'] = np.array(fpex.get_dicrotic_notch(peak, onsets))

    det_dn = np.array(fpex.get_dicrotic_notch(peak, onsets))
    vpg_fp = fpex.get_vpg_fiducials(onsets)
    apg_fp = fpex.get_apg_fiducials(onsets, peak)
    jpg_fp = fpex.get_jpg_fiducials(onsets, apg_fp)
    ppg_fp['dp'] = fpex.get_diastolic_peak(onsets, ppg_fp.dn, apg_fp.e)

    ppg_fp['on'] = onsets[0]
    ppg_fp['off'] = onsets[1]
    ppg_fp['sp'] = peak[0]

    # Merge fiducials
    det_fp = merge_fiducials(ppg_fp, vpg_fp, apg_fp, jpg_fp)

    return det_fp

# -----------------------------------------------
# Biomarker list
# -----------------------------------------------
biomarkers_lst = [
                    ["Tpi",   "Pulse interval, the time between the pulse onset and pulse offset", "[s]"],
                    ["Tsys",  "Systolic time, the time between the pulse onset and dicrotic notch", "[s]"],
                    ["Tdia",  "Diastolic time, the time is between the dicrotic notch and pulse offset", "[s]"],
                    ["Tsp",   "Systolic peak time, the time between the pulse onset and systolic peak", "[s]"],
                    ["Tdp",	  "Diastolic peak time, the time between the pulse onset and diastolic peak", "[s]"],
                    ["deltaT","Time delay, the time between the systolic peak and diastolic peak", "[s]"],
                    ["Tsw10", "Systolic width, the width at 10% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw25", "Systolic width, the width at 25% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw33", "Systolic width, the width at 33% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw50", "Systolic width, the width at 50% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw66", "Systolic width, the width at 66% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw75", "Systolic width, the width at 75% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tsw90", "Systolic width, the width at 90% of the systolic peak amplitude between the pulse onset and systolic peak", "[s]"],
                    ["Tdw10", "Diastolic width, the width at 10% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw25", "Diastolic width, the width at 25% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw33", "Diastolic width, the width at 33% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw50", "Diastolic width, the width at 50% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw66", "Diastolic width, the width at 66% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw75", "Diastolic width, the width at 75% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tdw90", "Diastolic width, the width at 90% of the systolic peak amplitude between the systolic peak and pulse offset", "[s]"],
                    ["Tpw10", "Pulse width, the sum of the systolic width and the diastolic width at 10%", "[s]"],
                    ["Tpw25", "Pulse width, the sum of the systolic width and the diastolic width at 25%", "[s]"],
                    ["Tpw33", "Pulse width, the sum of the systolic width and the diastolic width at 33%", "[s]"],
                    ["Tpw50", "Pulse width, the sum of the systolic width and the diastolic width at 50%", "[s]"],
                    ["Tpw66", "Pulse width, the sum of the systolic width and the diastolic width at 66%", "[s]"],
                    ["Tpw75", "Pulse width, the sum of the systolic width and the diastolic width at 75%", "[s]"],
                    ["Tpw90", "Pulse width, the sum of the systolic width and the diastolic width at 90%", "[s]"],
                    ["Asp",   "Systolic peak amplitude, the difference in amplitude between the pulse onset and systolic peak", "[nu]"],
                    ["Adn",   "Dicrotic notch amplitude, the difference in amplitude between the pulse onset and dicrotic notch", "[nu]"],
                    ["Adp",   "Diastolic peak amplitude, the difference in amplitude between the pulse onset and diastolic peak", "[nu]"],
                    ["Aoff",  "Pulse onset amplitude, the difference in amplitude between the pulse onset and pulse offset", "[nu]"],
                    ["AUCpi", "Area under pulse interval curve, the area under the pulse wave between pulse onset and pulse offset", "[nu]"],
                    ["AUCsys","Area under systolic curve, the area under the pulse wave between the pulse onset and the dicrotic notch", "[nu]"],
                    ["AUCdia","Area under diastolic curve, the area under the pulse wave between the dicrotic notch and pulse offset", "[nu]"],
                    ["Tsys/Tdia",    "Ratio of the systolic time vs. the diastolic time", "[%]"],
                    ["Tpw25/Tpi",    "Ratio of the pulse width at 25% of the systolic peak amplitude vs. the pulse interval", "[%]"],
                    ["Tpw50/Tpi",    "Ratio of the pulse width at 50% of the systolic peak amplitude vs. the pulse interval", "[%]"],
                    ["Tpw75/Tpi",    "Ratio of the pulse width at 75% of the systolic peak amplitude vs. the pulse interval", "[%]"],
                    ["Tpw25/Tsp",    "Ratio of the pulse width at 25% of the systolic peak amplitude vs. the systolic peak time", "[%]"],
                    ["Tpw50/Tsp",    "Ratio of the pulse width at 50% of the systolic peak amplitude vs. the systolic peak time", "[%]"],
                    ["Tpw75/Tsp",    "Ratio of the pulse width at 75% of the systolic peak amplitude vs. the systolic peak time", "[%]"],
                    ["Tdw10/Tsw10",  "Ratio of the diastolic width vs. the systolic width at 10% width", "[%]"],
                    ["Tdw25/Tsw25",  "Ratio of the diastolic width vs. the systolic width at 25% width", "[%]"],
                    ["Tdw33/Tsw33",  "Ratio of the diastolic width vs. the systolic width at 33% width", "[%]"],
                    ["Tdw50/Tsw50",  "Ratio of the diastolic width vs. the systolic width at 50% width", "[%]"],
                    ["Tdw66/Tsw66",  "Ratio of the diastolic width vs. the systolic width at 66% width", "[%]"],
                    ["Tdw75/Tsw75",  "Ratio of the diastolic width vs. the systolic width at 75% width", "[%]"],
                    ["Tdw90/Tsw90",  "Ratio of the diastolic width vs. the systolic width at 90% width", "[%]"],
                    ["Tsp/Tpi",      "Ratio of the systolic peak time vs. the pulse interval", "[%]"],
                    ["Asp/Aoff",     "Ratio of the systolic peak amplitude vs. the pulse offset amplitude", "[%]"],
                    ["Adp/Asp",      "Reflection index, the ratio of the diastolic peak amplitude vs. the systolic peak amplitude", "[%]"],
                    ["IPA",          "Inflection point area, the ratio of the area under diastolic curve vs. the area under systolic curve", "[nu]"],
                    ["Tsp/Asp",      "Ratio of the systolic peak time vs. the systolic peak amplitude", "[nu]"],
                    ["Asp/deltaT",   "Stiffness index, the ratio of the systolic peak amplitude vs. the time delay", "[nu]"],
                    ["Asp/(Tpi-Tsp)","Ratio of the systolic peak amplitude vs. the difference between the pulse interval and systolic peak time ", "[nu]"],
                    ["Tu",       "u-point time, the time between the pulse onset and u-point", "[s]"],
                    ["Tv",       "v-point time, the time between the pulse onset and v-point", "[s]"],
                    ["Tw",       "w-point time, the time between the pulse onset and w-point", "[s]"],
                    ["Ta",       "a-point time, the time between the pulse onset and a-point", "[s]"],
                    ["Tb",       "b-point time, the time between the pulse onset and b-point", "[s]"],
                    ["Tc",       "c-point time, the time between the pulse onset and c-point", "[s]"],
                    ["Td",       "d-point time, the time between the pulse onset and d-point", "[s]"],
                    ["Te",       "e-point time, the time between the pulse onset and e-point", "[s]"],
                    ["Tf",       "f-point time, the time between the pulse onset and f-point", "[s]"],
                    ["Tb-c",	 "b-c time, the time between the b-point and c-point", "[s]"],
                    ["Tb-d",	 "b-d time, the time between the b-point and d-point", "[s]"],
                    ["Tp1",	     "p1-point time, the time between the pulse onset and p1-point", "[s]"],
                    ["Tp2",      "p2-point time, the time between the pulse onset and p2-point", "[s]"],
                    ["Tp1-dp",   "p1-dia time, the time between the p1-point and diastolic peak", "[s]"],
                    ["Tp2-dp",   "p2-dia time, the time between the p2-point and diastolic peak", "[s]"],
                    # ["Tu/Tpi",       "Ratio of the u-point time vs. the pulse interval", "[%]"],
                    # ["Tv/Tpi",       "Ratio of the v-point time vs. the pulse interval", "[%]"],
                    # ["Tw/Tpi",       "Ratio of the w-point time vs. the pulse interval", "[%]"],
                    # ["Ta/Tpi",       "Ratio of the a-point time vs. the pulse interval", "[%]"],
                    # ["Tb/Tpi",       "Ratio of the b-point time vs. the pulse interval", "[%]"],
                    # ["Tc/Tpi",       "Ratio of the c-point time vs. the pulse interval", "[%]"],
                    # ["Td/Tpi",       "Ratio of the d-point time vs. the pulse interval", "[%]"],
                    # ["Te/Tpi",       "Ratio of the e-point time vs. the pulse interval", "[%]"],
                    # ["Tf/Tpi",       "Ratio of the f-point time vs. the pulse interval", "[%]"],
                    ["(Tu-Ta)/Tpi",  "Ratio of the difference between the u-point time and a-point time vs. the pulse interval", "[%]"],
                    ["(Tv-Tb)/Tpi",  "Ratio of the difference between the v-point time and b-point time vs. the pulse interval", "[%]"],
                    ["Au/Asp",       "Ratio of the u-point amplitude vs. the systolic peak amplitude", "[%]"],
                    ["Av/Au",        "Ratio of the v-point amplitude vs. the u-point amplitude", "[%]"],
                    ["Aw/Au",        "Ratio of the w-point amplitude vs. the u-point amplitude", "[%]"],
                    ["Ab/Aa",        "Ratio of the b-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ac/Aa",        "Ratio of the c-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ad/Aa",        "Ratio of the d-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ae/Aa",        "Ratio of the e-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Af/Aa",        "Ratio of the f-point amplitude vs. the a-point amplitude", "[%]"],
                    ["Ap2/Ap1",      "Ratio of the p2-point amplitude vs. the p1-point amplitude", "[%]"],
                    ["(Ac-Ab)/Aa",   "Ratio of the difference between the b-point amplitude and c-point amplitude vs. the a-point amplitude", "[%]"],
                    ["(Ad-Ab)/Aa",   "Ratio of the difference between the b-point amplitude and d-point amplitude vs. the a-point amplitude", "[%]"],
                    ["AGI",          "Aging Index, (Ab-Ac-Ad-Ae)/Aa", "[%]"],
                    ["AGImod",       "Modified aging index, (Ab-Ac-Ad)/Aa", "[%]"],
                    ["AGIinf",       "Informal aging index, (Ab-Ae)/Aa", "[%]"],
                    ["AI",           "Augmentation index, (PPG(Tp2)-PPG(Tp1))/Asp", "[%]"],
                    ["RIp1",         "Reflection index of p1, Adp/(PPG(Tp1)-PPG(Tpi(0)))", "[%]"],
                    ["RIp2",         "Reflection index of p2, Adp/(PPG(p2)-PPG(Tpi(0)))", "[%]"],
                    ["SC",           "Spring constant, PPG''(Tsp)/((Asp-Au)/Asp)", "[nu]"],
                    ["IPAD",         "Inflection point area plus normalised d-point amplitude, AUCdia/AUCsys+Ad/Aa", "[nu]"],
    ]

header = ['name', 'definition', 'unit']
biomarkers_lst = pd.DataFrame(biomarkers_lst, columns=header)

# -----------------------------------------------
# Find biomarkers in a PPG beat
# -----------------------------------------------
def find_biomarkers(s,fp):
    fs=s.fs
    ppg=s.ppg
    data = DotMap()

    df = pd.DataFrame(columns=['onset','offset','peak'])
    df_biomarkers = pd.DataFrame(columns=biomarkers_lst)
    peaks = fp.sp.values
    onsets = fp.on.values
    offsets = fp.off.values

    onset = onsets[0]
    offset = offsets[0]
    data.ppg = ppg[int(onset):int(offset)]
    data.vpg = s.vpg[int(onset):int(offset)]
    data.apg = s.apg[int(onset):int(offset)]
    data.jpg = s.jpg[int(onset):int(offset)]

    temp_fiducials = fp

    peak = peaks[(peaks > onset) * (peaks < offset)]

    peak = peak[0]

    # temp_fiducials = fps_df[mask]
    peak_value = ppg[peak]
    peak_time = peak / fs
    onset_value = ppg[onset]
    onset_time = onset / fs

    offset_value = ppg[offset]
    offset_time = offset / fs

    idx_array = np.where(peaks == peak)
    idx = idx_array[0]
    onsets_values = np.array([onset_value, offset_value])
    onsets_times = np.array([onset_time, offset_time])

    biomarkers_extractor = BmExctator(data, peak_value, peak_time, np.nan, np.nan, onsets_values, onsets_times, fs, biomarkers_lst.name,temp_fiducials)
    biomarkers_vec = biomarkers_extractor.get_biomarker_extract_func()

    df_biomarkers = pd.DataFrame(columns=biomarkers_lst)
    lst = list(biomarkers_vec)
    df_biomarkers.loc[0] = lst
    return df_biomarkers

#-----------------------------------------------
# Detect PPG beats using the QPPG algorithm
# 
# Reference:
#   Vest, Adriana Nicholson, Giulia Da Poian, Qiao Li, Chengyu Liu, Shamim Nemati, 
#   Amit J. Shah and Gari D. CliQord. “An open source benchmarked toolbox for cardiovascular 
#   waveform and interval analysis.” Physiological Measurement 39 (2018)
# original reference:
#   Li, Q., Clifford, G.D. "Dynamic time warping and machine learning for signal quality    
#   assessment of pulsatile signals." Physiol Meas 33, 1491-1501 (2012).
#-----------------------------------------------
def qppg(data, fs=125, from_idx=0, to_idx=None):
    if to_idx is None:
        to_idx = len(data)
    
    global BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow

    idxPeaks = []
    beat_n = 0

    sps = fs  # Sampling Frequency

    BUFLN = 4096  # must be a power of 2, see slpsamp()
    EYE_CLS = 0.34  # eye-closing period is set to 0.34 sec (340 ms) for PPG
    LPERIOD = sps * 1  # learning period is the first LPERIOD samples
    SLPW = 0.17  # Slope width (170ms) for PPG
    NDP = 2.5  # adjust threshold if no pulse found in NDP seconds
    TmDEF = 5  # minimum threshold value (default)
    Tm = TmDEF

    BUFLN2 = BUFLN * 2

    INVALID_DATA = -32768
    if data[0] <= INVALID_DATA + 10:
        data[0] = np.mean(data)
    inv = np.where(data <= INVALID_DATA + 10)[0]
    for i in inv:
        data[i] = data[i - 1]

    # re-scale data to ~ +/- 2000
    if len(data) < 5 * 60 * sps:
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 4000 - 2000
    else:
        max_data = []
        min_data = []
        for i in range(0, len(data), 5 * 60 * sps):
            max_data.append(np.max(data[i:min(i + 5 * 60 * sps, len(data))]))
            min_data.append(np.min(data[i:min(i + 5 * 60 * sps, len(data))]))
        data = (data - np.median(min_data)) / (np.median(max_data) - np.median(min_data)) * 4000 - 2000

    samplingInterval = 1000.0 / sps
    spm = 60 * sps
    EyeClosing = round(sps * EYE_CLS)  # set eye-closing period
    ExpectPeriod = round(sps * NDP)  # maximum expected RR interval
    SLPwindow = round(sps * SLPW)  # slope window size
    timer = 0

    ebuf = np.zeros(BUFLN)
    lbuf = ebuf.copy()
    if from_idx > BUFLN:
        tt_2 = from_idx - BUFLN
    else:
        tt_2 = 0
    aet = 0

    t1 = 8 * sps
    t1 += from_idx
    T0 = 0
    n = 0
    for t in range(from_idx, t1):
        temp = slpsamp(t, data)
        if temp > INVALID_DATA + 10:
            T0 += temp
            n += 1
    T0 /= n
    Ta = 3 * T0

    learning = True
    t = from_idx

    # Main loop
    while t <= to_idx:
        if learning:
            if t > from_idx + LPERIOD:
                learning = False
                T1 = T0
                t = from_idx  # start over
            else:
                T1 = 2 * T0

        temp = slpsamp(t, data)

        if temp > T1:  # found a possible ABP pulse near t
            timer = 0  # used for counting the time after previous ABP pulse
            maxd = temp
            mind = maxd
            tmax = t
            for tt in range(t + 1, t + EyeClosing):
                temp2 = slpsamp(tt, data)
                if temp2 > maxd:
                    maxd = temp2
                    tmax = tt
            if maxd == temp:
                t += 1
                continue

            for tt in range(tmax, t - EyeClosing // 2, -1):
                temp2 = slpsamp(tt, data)
                if temp2 < mind:
                    mind = temp2
            if maxd > mind + 10:
                onset = (maxd - mind) / 100 + 2
                tpq = t - round(0.04 * fs)
                maxmin_2_3_threshold = (maxd - mind) * 2.0 / 3
                for tt in range(tmax, t - EyeClosing // 2, -1):
                    temp2 = slpsamp(tt, data)
                    if temp2 < maxmin_2_3_threshold:
                        break
                for tt in range(tt, t - EyeClosing // 2 + round(0.024 * fs), -1):
                    temp2 = slpsamp(tt, data)
                    temp3 = slpsamp(tt - round(0.024 * fs), data)
                    if temp2 - temp3 < onset:
                        tpq = tt - round(0.016 * fs)
                        break

                # find valley from the original data around 0.25s of tpq
                valley_v = round(tpq)
                for valley_i in range(round(max(1, tpq - round(0.20 * fs))), round(min(tpq + round(0.05 * fs), len(data) - 1))):
                    if valley_v <= 0:
                        t += 1
                        continue

                    if data[valley_v] > data[valley_i] and data[valley_i] <= data[valley_i - 1] and data[valley_i] <= data[valley_i + 1]:
                        valley_v = valley_i

                if not learning:
                    if beat_n == 0:
                        if round(valley_v) >= 0:
                            idxPeaks.append(round(valley_v))
                            beat_n += 1
                    else:
                        if round(valley_v) > idxPeaks[beat_n - 1]:
                            idxPeaks.append(round(valley_v))
                            beat_n += 1

                # Adjust thresholds
                Ta += (maxd - Ta) / 10
                T1 = Ta / 3

                # Lock out further detections during the eye-closing period
                t = tpq + EyeClosing
        else:
            if not learning:
                timer += 1
                if timer > ExpectPeriod and Ta > Tm:
                    Ta -= 1
                    T1 = Ta / 3

        t += 1

    return idxPeaks

def slpsamp(t, data):
    global BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow

    while t > tt_2:
        prevVal = 0

        if tt_2 > 0 and tt_2 - 1 >= 0 and tt_2 < len(data) and tt_2 - 1 < len(data):
            val2 = data[tt_2 - 1]
            val1 = data[tt_2]
        else:
            val2 = prevVal
            val1 = val2
        prevVal = val2
        dy = val1 - val2
        if dy < 0:
            dy = 0
        tt_2 += 1
        M = round(tt_2 % (BUFLN - 1))
        et = dy
        ebuf[M] = et
        aet = 0
        for i in range(SLPwindow):
            p = M - i
            if p < 0:
                p += BUFLN
            aet += ebuf[p]
        lbuf[M] = aet

    M3 = round(t % (BUFLN - 1))
    return lbuf[M3]