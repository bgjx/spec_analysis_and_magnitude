#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:32:03 2022

@author : ARHAM ZAKKI EDELO
@contact: edelo.arham@gmail.com
"""
from obspy import UTCDateTime, Stream, read, Trace, read_inventory
from obspy.signal import rotate
from obspy.geodetics import gps2dist_azimuth
import numpy as np
import os, glob, subprocess, sys
from pathlib import PurePath, Path
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd 
from scipy import signal
from loguru import logger 

print('''
Python code to calculate moment magnitude

Before you run this program, make sure you have changed all the path correctly.      
      ''')

# Global Parameters
#============================================================================================

# instrument correction parameters
water_level = 20                      # water level 
pre_filter = [0.1, 0.25, 247.25, 250] # these values need to be customized for a specific bandwidth

# plotting parameters
# static window parameter 
#time_after_pick_p = 0.45
#time_after_pick_s = 1.2

# time padding and position parameters
time_before_pick = 0.1

# noise spectra duration before P pick
noise_time = 0.5
noise_padding = 0.2

# setting frequency range for spectral fitting (default spectral f-band frequency range)
f_min = 0.75
f_max = 25

# List of functions 
#============================================================================================

# # rotate functions
# def rotate_component(st, azim): ## rotation function
    # # make empty stream
    # sst_rotated = Stream()
    
    # # select the trace of respected component and the status
    # tr_Z = st.select(component='Z')[0]
    
    # tr_N = st.select(component='N')[0]
    # tr_T_status = tr_N.stats
    
    # tr_E = st.select(component='E')[0]
    # tr_R_status = tr_E.stats
    
    # # do the rotation
    # tr_R,tr_T = rotate.rotate_ne_rt(tr_N.data, tr_E.data, azim)
    
    # # convert numpy ndarray to trace
    # tr_R = Trace(tr_R)
    # tr_T = Trace(tr_T)
    
    # # update the trace status
    # tr_R.stats.update(tr_R_status)
    # tr_R.stats.component = 'R'
    # tr_T.stats.update(tr_T_status)
    # tr_T.stats.component = 'T'
    
    # # adding stream
    # sst_rotated.extend([tr_Z, tr_R, tr_T])
    # return sst_rotated

# intrument response remove function
def instrument_remove (st, calibration_path, fig_path, fig_statement = False):
    st_removed=Stream()
    for tr in st:
        try:
            
            # read the calibration file
            sta, comp = tr.stats.station, tr.stats.component
            inv_path=calibration_path.joinpath(f"RESP.ML.{sta}..BH{comp}")
            inv=read_inventory(inv_path, format='RESP')
  
            # remove response, be cautious with the water level parameter!
            if fig_statement:
                rtr=tr.remove_response(inventory=inv, pre_filt=pre_filter,water_level=water_level,output='DISP',zero_mean=True, taper=True, taper_fraction=0.05, plot = fig_path.joinpath(f"fig_{sta}_BH{comp}"))
            else:
                rtr=tr.remove_response(inventory=inv, pre_filt=pre_filter,water_level=water_level,output='DISP',zero_mean=True, taper=True, taper_fraction=0.05, plot=False )
            
            ## re-detrending
            rtr.detrend("linear")
            
            st_removed+=rtr
        except Exception as e:
            print(e)
            pass
            
    return st_removed

# windowing functions
def window_trace(tr, P_arr, S_arr):
    # dinamic window parameter
    s_p_time = float(S_arr - P_arr)    
    time_after_pick_p = 0.75 * s_p_time
    time_after_pick_s = 1.50 * s_p_time
        
    # determine the data index for windowing P phases
    p_phase_start_index = int(round( (P_arr - tr.stats.starttime )/ tr.stats.delta,4)) - \
                            int(round(time_before_pick  / tr.stats.delta,4))
    p_phase_finish_index = int(round((P_arr - tr.stats.starttime )/ tr.stats.delta,4))+ \
                            int(round(time_after_pick_p / tr.stats.delta,4))
    
    # determine the data index for windowing S phases
    s_phase_start_index = int(round( (S_arr - tr.stats.starttime )/ tr.stats.delta,4))- \
                            int(round(time_before_pick / tr.stats.delta,4))
    s_phase_finish_index = int(round((S_arr - tr.stats.starttime )/ tr.stats.delta,4))+ \
                            int(round(time_after_pick_s / tr.stats.delta,4))

    # determine the data index for windowing Noise
    noise_start_index = int(round( (P_arr - tr.stats.starttime )/ tr.stats.delta,4)) - \
                            int(round( noise_time / tr.stats.delta,4))
    noise_finish_index  = int(round( (P_arr - tr.stats.starttime )/ tr.stats.delta,4)) - \
                            int(round( noise_padding / tr.stats.delta,4)) 
    
    # windowing the data for fitting spectrum and modelling
    P_data     = tr.data[p_phase_start_index : p_phase_finish_index + 1]
    S_data     = tr.data[s_phase_start_index : s_phase_finish_index + 1]
    noise_data = tr.data[noise_start_index : noise_finish_index + 1]
    return P_data, S_data, noise_data
    
# function to check the trace SNR
def trace_snr (data, noise):
    # testing snr by the RMS ratio
    data_rms = np.sqrt(np.mean(data**2))
    noise_rms = np.sqrt(np.mean(noise**2))
    return data_rms / noise_rms
   
# fucntion for calculating the power spectral density (PSD)
def calculate_spectra(trace_data, sampling_rate):
    frequency, power_spectra = signal.welch(trace_data, sampling_rate, nperseg = len(trace_data))
    return frequency, power_spectra

# functiton for spectrum windowing within spesific f_band
def window_band(frequencies, spectrums, f_min, f_max):
    indices = np.where((frequencies >= f_min) & (frequencies <= f_max))
    freq = frequencies[indices]
    spec = spectrums[indices]
    return  freq, spec
    
# fitting spectrum using Grid search algorithm
def fit_spectrum (frequencies, spectrums, traveltime, f_min, f_max):
    # windowing frequencies and spectrum within f band    
    freq, spectrum = window_band(frequencies, spectrums, f_min, f_max)
    
    # setting initial guess
    peak_omega = spectrum.max()
    omega_0 = np.linspace(peak_omega/20, peak_omega*20, 100)
    Q_factor = np.linspace(50, 2500, 50)
    f_c = np.linspace(0.75, 25, 50)
    
    # rms and error handler
    rms_e = None
    error = np.inf
    
    # define callable function
    def f(freqs, omega, qfactor, f_cor):
        return calculate_source_spectrum(freqs, omega, qfactor, f_cor, traveltime)
        
    # start guessing
    for i in range(len(omega_0)):
        for j in range(len(Q_factor)):
            for k in range(len(f_c)):
                fwd = f(freq, omega_0[i], Q_factor[j], f_c[k])
                rms_e = np.sqrt(np.mean((fwd - spectrum)**2))
                if rms_e < error:
                    error = rms_e
                    omega_0_fit = omega_0[i]
                    Q_factor_fit = Q_factor[j]
                    f_c_fit = f_c[k]
                    
    # calculate the fitted power spectral density from tuned parameter
    x_tuned = np.linspace(0.75, 100, 100)
    y_tuned = f(x_tuned, omega_0_fit, Q_factor_fit, f_c_fit) 
                    
    return omega_0_fit, Q_factor_fit, f_c_fit, rms_e, x_tuned, y_tuned
    
    
# # fitting spectrum function using levenberg-marquardt algorithm
# def fit_spectrum_first(spectrum, frequencies, traveltime, initial_omega_0,
    # initial_f_c):
    # """
    # Fit a theoretical source spectrum to a measured source spectrum.
    # Uses a Levenburg-Marquardt algorithm.
    # :param spectrum: The measured source spectrum.
    # :param frequencies: The corresponding frequencies.
    # :para traveltime: Event traveltime in [s].
    # :param initial_omega_0: Initial guess for Omega_0.
    # :param initial_f_c: Initial guess for the corner frequency.
    # :param initial_q: initial quality factor
    # :returns: Best fits and standard deviations.
        # (Omega_0, f_c, Omega_0_std, f_c_std)
        # Returns None, if the fit failed.
    # """
    # def f(frequencies, omega_0, f_c):
        # return calculate_source_spectrum(frequencies, omega_0, f_c,
                # Qfactor, traveltime)
    # popt, pcov = scipy.optimize.curve_fit(f, frequencies, spectrum, \
        # p0=list([initial_omega_0, initial_f_c]), maxfev=100000)        # maxfev is the maximum number of function calls allowed during the optimization
    # # p0 is the initial guest that will be optimized by the fit method
    # # popt is the optimezed parameters and the pcov is the covariance matrix
    
    # x_fit=frequencies
    # y_fit= f(x_fit, *popt)
    
    # if popt is None:
        # return None
    # return popt[0], popt[1], pcov[0, 0], pcov[1, 1], x_fit,y_fit
 
# function for calculating the source spectrum (spectrum model) 
def calculate_source_spectrum(frequencies, omega_0, Q, corner_frequency, 
    traveltime):
    """
    After Abercrombie (1995) and Boatwright (1980).
    Abercrombie, R. E. (1995). Earthquake locations using single-station deep
    borehole recordings: Implications for microseismicity on the San Andreas
    fault in southern California. Journal of Geophysical Research, 100,
    24003–24013.
    Boatwright, J. (1980). A spectral theory for circular seismic sources,
    simple estimates of source dimension, dynamic stress drop, and radiated
    energy. Bulletin of the Seismological Society of America, 70(1).
    The used formula is:
        Omega(f) = (Omege(0) * e^(-pi * f * T / Q)) / (1 + (f/f_c)^4) ^ 0.5
    :param frequencies: Input array to perform the calculation on.
    :param omega_0: Low frequency amplitude in [meter x second].
    :param corner_frequency: Corner frequency in [Hz].
    :param Q: Quality factor.
    :param traveltime: Traveltime in [s].
    """
    num = omega_0 * np.exp(-np.pi * frequencies * traveltime / Q)
    denom = (1 + (frequencies / corner_frequency) ** 4)**0.5
    return num / denom

# function for calculating the moment magnitude
def calculate_moment_magnitude(wave_path, hypo_df, pick_df, station, calibration_path, id, fig_path, fig_statement = False):

    # create figure statement
    if fig_statement:
        try:
            fig, axs= plt.subplots((len(list(pick_df.get("Station")))*6),2, figsize=(20,140) ) # for plotting purposes
            plt.subplots_adjust(hspace=0.5)
            axs[0,0].set_title("P Phase Spectra Profile", fontsize='20')
            axs[0,1].set_title("S Phase Spectra Profile", fontsize='20')
            counter = 0
        except Exception as e:
            pass
    
    # internal prededined parameters 

    # radiaton pattern parameters
    r_pattern_P = 0.440
    r_pattern_S = 0.600
    
    # kappa parametter
    k_P = 0.32
    k_S = 0.21
    
    # velocity and density each layer parameters
    layer_top   = [ [-2,-0.3],[-0.3,0],[0, 1.2],[1.2, 6.1], [6.1, 14.1], [14.1,9999] ]
    velocity_Vp = [3.82, 4.00, 4.50, 4.60, 6.20, 8.00]                                      # km/s
    velocity_Vs = [2.30, 2.30, 2.53, 2.53, 3.44, 4.44]                                      # km/s
    density     = [ 2375.84, 2465.34, 2465.34, 2529.08, 2750.80, 2931.80]                   # kg/m3

    # Holder value for average moments, source radius, and corner_frequencies from several stations
    moments = []
    corner_frequencies = []
    source_radius = []
    
    # dict handler for fitting result detail
    fitting_result = {
        "ID":[],
        "Station":[],
        "Component":[],
        "F_corner_P":[],
        "F_corner_S":[],
        "Qfactor_P":[],
        "Qfactor_S":[],
        "Omega_0_P(nms)":[],
        "Omega_0_S(nms)":[],
        "RMS_e_P(nms)":[],
        "RMS_e_S(nms)":[]
    }

    # find the origin time, latitude, longitude and the depth of the event from hypo data
    origin_time = UTCDateTime(f"{hypo_df.Year.iloc[0]}-{int(hypo_df.Month.iloc[0]):02d}-{int(hypo_df.Day.iloc[0]):02d}T{int(hypo_df.Hour.iloc[0]):02d}:{int(hypo_df.Minute.iloc[0]):02d}:{float(hypo_df.T0.iloc[0]):012.9f}") 
    hypo_lat, hypo_lon , hypo_depth =  hypo_df.Lat.iloc[0], hypo_df.Lon.iloc[0], hypo_df.Depth.iloc[0]
    
    # find the correct velocity and density value for the spesific layer depth
    for layer in range(len(layer_top)):
        top_layer_limit = layer_top[layer][0]
        bottom_layer_limit = layer_top[layer][1]
        if (top_layer_limit*1000)   <= hypo_depth <= (bottom_layer_limit*1000):
            velocity_P = velocity_Vp[layer]*1000  # velocity in m/s
            velocity_S = velocity_Vs[layer]*1000  # velocity in m/s
            density_value = density[layer]
        else:
            pass

    # read waveform
    try:
        stream = read(wave_path.joinpath(f"{id}\\*"))
    except Exception as e:
        logger.exception("An error occured during runtime:", str(e))
        pass
        
    st = stream.copy()
    
    # start spectrum fitting and magnitude estimation
    for sta in list(pick_df.get("Station")):
        
        # get the station coordinat
        sta_xyz = station[station.Stations == sta]
        sta_lat, sta_lon, sta_elev = sta_xyz.Lat.iloc[0], sta_xyz.Lon.iloc[0], sta_xyz.Elev.iloc[0]
        
        # calculate the source distance and the azimuth (hypo to station azimuth)
        epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
        source_distance = np.sqrt(epicentral_distance**2 + (hypo_depth + sta_elev)**2)

        # get the pick_df data for P arrival and S arrival 
        year, month, day, hour  = pick_df.Year[pick_df.Station == sta].iloc[0], pick_df.Month[pick_df.Station == sta].iloc[0], pick_df.Day[pick_df.Station == sta].iloc[0], pick_df.Hour[pick_df.Station == sta].iloc[0]
        minute_P, sec_P         = pick_df.Minutes_P[pick_df.Station == sta].iloc[0], pick_df.P_Arr_Sec[pick_df.Station == sta].iloc[0]
        minute_S, sec_S         = pick_df.Minutes_S[pick_df.Station == sta].iloc[0], pick_df.S_Arr_Sec[pick_df.Station == sta].iloc[0]
        P_pick_time             = UTCDateTime(f"{year}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute_P):02d}:{float(sec_P):012.9f}")
        S_pick_time             = UTCDateTime(f"{year}-{int(month):02d}-{int(day):02d}T{int(hour):02d}:{int(minute_S):02d}:{float(sec_S):012.9f}")
        
        # object holder value for omegas and corner_freq from 3 component of each station
        omegas_P = []
        omegas_S = []
        corner_freqs_P = []
        corner_freqs_S = []

        # select the respected seismogram from the stream
        st2 = st.select(station = sta)
        
        # check if all 3 component complete
        if len(st2) < 3:
            logger.exception("An error occured during runtime:", str(e))
            continue
        else:
            # if 3 components are complete, do the instrument response removal
            st_removed = instrument_remove(st2, calibration_path, fig_path)
            
            # do the component rotation from NE to RT
            #st_rotated = rotate_component(st_removed, azimuth)
            
        # extract the three component traces from the rotated stream
        [tr_Z, tr_N, tr_E] = [st_removed.select(component = comp)[0] for comp in ['Z' ,'N', 'E'] ]
        
        for tr in [tr_Z, tr_N, tr_E]:
        
            # window the trace
            p_window_data, s_window_data, noise_window_data = window_trace(tr, P_pick_time, S_pick_time)
            
            # check the signal SNR
            p_snr = trace_snr( p_window_data, noise_window_data)
            s_snr = trace_snr( s_window_data, noise_window_data)
            if p_snr < 1 and s_snr < 1:
                logger.exception("An error occured during runtime:", str(e))
                continue
                
            # get the sampling rate
            fs = 1/tr.stats.delta
            
            # calculate source spectrum
            freq_P, spec_P = calculate_spectra(p_window_data, fs)
            freq_S, spec_S = calculate_spectra(s_window_data, fs)

            # calculate the noise spectra
            freq_N, spec_N = calculate_spectra(noise_window_data, fs)
            
            # fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using grid search algorithm
            try:
                fit_P = fit_spectrum(freq_P, spec_P, abs(float(P_pick_time - origin_time)), f_min, f_max)  # 10 for corner frequency initial guess, and 1200 for Q initial guess
                fit_S = fit_spectrum(freq_S, spec_S, abs(float(S_pick_time - origin_time)), f_min, f_max)  # 10 for corner frequency initial guess, and 1200 for Q initial guess
            except Exception as e:
                logger.exception("An error occured during runtime:", str(e))
                continue
                
            # # fitting the spectrum, find the optimal value of Omega_O, corner frequency and Q using levenberg-marquardt algorithm
            # try:
                # fit_P = fit_spectrum_first(spec_P, freq_P, float(P_pick_time - origin_time),spec_P.max(), 10.0)  # 10 for corner frequency initial guess, and 1200 for Q initial guess
                # fit_S = fit_spectrum_first(spec_S, freq_S, float(S_pick_time - origin_time),spec_S.max(), 10.0)  # 10 for corner frequency initial guess, and 1200 for Q initial guess
            # except Exception as e:
                # print("Issue 1 :",e)
                # continue

            if fit_P is None and fit_S is None:
                continue

            # fitting spectrum output
            Omega_0_P, Q_factor_p, f_c_P, err_P, x_fit_P, y_fit_P = fit_P
            Omega_0_S, Q_factor_S, f_c_S, err_S, x_fit_S, y_fit_S = fit_S
            
            # append the fitting spectrum output to the holder list
            Omega_0_P = np.sqrt(Omega_0_P)
            Omega_0_S = np.sqrt(Omega_0_S)
            
            err_P = np.sqrt(err_P)
            err_S = np.sqrt(err_S)
            
            
            # append the omegas 
            omegas_P.append(Omega_0_P)
            omegas_S.append(Omega_0_S)
            
            # append the corner frequency
            corner_freqs_P.append(f_c_P)
            corner_freqs_S.append(f_c_S)
            
            # updating the fitting dict handler 
            fitting_result["ID"].append(id)
            fitting_result["Station"].append(sta)
            fitting_result["Component"].append(tr.stats.component)
            fitting_result["F_corner_P"].append(f_c_P)
            fitting_result["F_corner_S"].append(f_c_S)
            fitting_result["Qfactor_P"].append(Q_factor_p)
            fitting_result["Qfactor_S"].append(Q_factor_S)
            fitting_result["Omega_0_P(nms)"].append((Omega_0_P*1e9))
            fitting_result["Omega_0_S(nms)"].append((Omega_0_S*1e9))
            fitting_result["RMS_e_P(nms)"].append((err_P*1e9))
            fitting_result["RMS_e_S(nms)"].append((err_S*1e9))

            # create figure
            if fig_statement:
                # frequency window for plotting purposes
                f_min_plot = 0.75
                f_max_plot = 100
                freq_P, spec_P = window_band(freq_P, spec_P, f_min_plot, f_max_plot)
                freq_S, spec_S = window_band(freq_S, spec_S, f_min_plot, f_max_plot)
                freq_N, spec_N = window_band(freq_N, spec_N, f_min_plot, f_max_plot)

                # dinamic window parameter
                s_p_time = float(S_pick_time - P_pick_time)    
                time_after_pick_p = 0.80 * s_p_time
                time_after_pick_s = 1.75 * s_p_time
                
                try:
                    # windowing trace data to be displayed
                    tr_d = tr.copy()
                    start_time = tr_d.stats.starttime
                    before = (P_pick_time - start_time) - 2.0
                    after = (S_pick_time - start_time) + 6.0
                    tr_d.trim(start_time+before, start_time+after)
                    start_time2 = tr_d.stats.starttime
                    station_plot = tr_d.stats.station
                    component_plot = tr_d.stats.component

                    # plot for P phase
                    axs[counter][0].plot(tr_d.times(), tr_d.data, 'k')
                    axs[counter][0].axvline( x= (P_pick_time - start_time2 ), color='r', linestyle='-', label='P arrival')
                    axs[counter][0].axvline( x= (S_pick_time - start_time2 ), color='b', linestyle='-', label='S arrival')
                    axs[counter][0].axvline( x= (P_pick_time - time_before_pick -  start_time2), color='g', linestyle='--')
                    axs[counter][0].axvline( x= (P_pick_time + time_after_pick_p - start_time2), color='g', linestyle='--', label='P phase window')
                    axs[counter][0].axvline( x= (P_pick_time - noise_time -  start_time2), color='gray', linestyle='--')
                    axs[counter][0].axvline( x= (P_pick_time - noise_padding  - start_time2), color='gray', linestyle='--', label='Noise window')
                    axs[counter][0].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
                    axs[counter][0].legend()
                    axs[counter][0].set_xlabel("Relative Time (s)")
                    axs[counter][0].set_ylabel("Amp (m)")
                   
                    # for s phase
                    axs[counter][1].plot(tr_d.times(), tr_d.data, 'k')
                    axs[counter][1].axvline( x= (P_pick_time - start_time2 ), color='r', linestyle='-', label='P arrival')
                    axs[counter][1].axvline( x= (S_pick_time - start_time2), color='b', linestyle='-', label='S arrival')
                    axs[counter][1].axvline( x= (S_pick_time - time_before_pick -  start_time2  ), color='g', linestyle='--')
                    axs[counter][1].axvline( x= (S_pick_time + time_after_pick_s - start_time2 ), color='g', linestyle='--', label='S phase window')
                    axs[counter][1].axvline( x= (P_pick_time - noise_time -  start_time2), color='gray', linestyle='--')
                    axs[counter][1].axvline( x= (P_pick_time - noise_padding  - start_time2), color='gray', linestyle='--', label='Noise window')
                    axs[counter][1].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
                    axs[counter][1].legend()
                    axs[counter][1].set_xlabel("Relative Time (s)")
                    axs[counter][1].set_ylabel("Amp (m)")
                   
                    # plot the spectra (P, S dan Noise spectra)
                    counter+=1
                   
                    axs[counter][0].loglog(freq_P, spec_P, color='black', label='P spectra')
                    axs[counter][0].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
                    axs[counter][0].loglog(x_fit_P, y_fit_P, 'b-', label='Fitted P Spectra')
                    axs[counter][0].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
                    axs[counter][0].legend()
                    axs[counter][0].set_xlabel("Frequencies (Hz)")
                    axs[counter][0].set_ylabel("Amp (m/Hz)")
                   
                    axs[counter][1].loglog(freq_S, spec_S, color='black', label='S spectra')
                    axs[counter][1].loglog(freq_N, spec_N, color='gray', label='Noise spectra')
                    axs[counter][1].loglog(x_fit_S, y_fit_S, 'b-', label='Fitted S Spectra')
                    axs[counter][1].set_title("{}_BH{}".format(station_plot, component_plot), loc="right",va='center')
                    axs[counter][1].legend()
                    axs[counter][1].set_xlabel("Frequencies (Hz)")
                    axs[counter][1].set_ylabel("Amp (m/Hz)")
                   
                    counter +=1
               
                except Exception as e:
                    logger.exception("An error occured during runtime:", str(e))
                    pass
                    
        # calculate the moment magnitude
        try:
            # calculate the  resultant omega
            omega_P_rst = np.sum((np.array(omegas_P))**2)
            omega_S_rst = np.sum((np.array(omegas_S))**2)
         
            ## calculate seismic moment
            M_0_P = 4.0 * np.pi * density_value * (velocity_P ** 3) * source_distance * \
                    np.sqrt(omega_P_rst) / \
                    (r_pattern_P)                                                   ### should it be multipled by 2 ??
                    
            M_0_S = 4.0 * np.pi * density_value * (velocity_S ** 3) * source_distance * \
                    np.sqrt(omega_S_rst) / \
                    (r_pattern_S)                                                    ### should it be multipled by 2 ??
            
            # calculate source radius
            r_P = 3 * k_P * velocity_P / sum(corner_freqs_P) # result in meter, times 3 because it is a three components
            r_S = 3 * k_S * velocity_S / sum(corner_freqs_S) # result in meter, times 3 because it is a three components
            
            # extend the moments object holder to calculate the moment magnitude
            moments.extend([M_0_P, M_0_S]) 
            
            # calculate corner frequency mean
            corner_freqs_P = np.array(corner_freqs_P)
            corner_freqs_S = np.array(corner_freqs_S)
            corner_freq_P = corner_freqs_P.mean()
            corner_freq_S = corner_freqs_S.mean()
            corner_frequencies.extend([corner_freq_P, corner_freq_S])
            
            # extend the source radius
            source_radius.extend([r_P, r_S])        
     
        except Exception as e:
            logger.exception("An error occured during runtime:", str(e))
            continue
            
    # Calculate the seismic moment viabnv  basic statistics.
    moments = np.array(moments)
    moment = moments.mean()
    moment_std = moments.std()
    
    ## calculate the corner frequencies via basic statistics.
    corner_frequencies = np.array(corner_frequencies)
    corner_frequency = corner_frequencies.mean()
    corner_frequency_std = corner_frequencies.std()

    # Calculate the source radius.
    source_radius = np.array(source_radius)
    source_rad = source_radius.mean()
    source_radius_std = source_radius.std()
    
    # Calculate the stress drop of the event based on the average moment and source radius
    stress_drop = ((7 * moment) / (16 * (source_rad * 0.001) ** 3))*1e-14
    stress_drop_std = np.sqrt((stress_drop ** 2) * (((moment_std ** 2) / (moment ** 2)) + \
    (9 * source_rad * source_radius_std ** 2)))   
        
    # Calculate the moment magnitude
    Mw  = ((2.0 / 3.0) * np.log10(moment)) - 6.07
    
    # calculate moment magnitude from seismic moment standard deviation 
    Mw_std = 2.0 / 3.0 * moment_std / (moment * np.log(10))
 
    results = {"ID":[f"{id}"], 
                "Fc":[f"{corner_frequency}"],
                "Fc_std":[f"{corner_frequency_std}"],
                "Mw":[f"{Mw}"],
                "Mw_std":[f"{Mw_std}"],
                "Src_rad(m)":[f"{source_rad}"],
                "Src_rad_std":[f"{source_radius_std}"],
                "Stress_drop(bar)":[f"{stress_drop}"],
                "Stress_drop_std":[f"{stress_drop_std}"]
                }
                
    if fig_statement : 
        fig.suptitle(f"Event {id} Spesctral Fitting Profile", fontsize='24', fontweight='bold')
        #plt.title("Event {} Spectral Fitting Profile".format(ID), fontsize='20')
        plt.savefig(fig_path.joinpath(f"event_{id}.png"))
    
    return results, fitting_result

# End of functions 
#============================================================================================

if __name__ == "__main__" :
    prompt=str(input('Please type yes/no if you had changed the path :'))
    if prompt != 'yes':
        sys.exit("Ok, please correct the path first!")
    else:
        print("Process the program ....\n\n")
        pass

    # setting logger for debugging 
    logger.add("runtime.log", level="ERROR", backtrace=True, diagnose=True)
    
    # initialize input and output path
    wave_path       = Path(r"F:\SEML\DATA TRIMMING\EVENT DATA TRIM\2023\2023 09")                           # trimmed waveform location
    hypo_input      = Path(r"F:\SEML\CATALOG HYPOCENTER\catalog\hypo_reloc.xlsx")                           # relocated catalog
    sta_input       = Path(r"F:\SEML\STATION AND VELOCITY DATA\SEML_station.xlsx")                          # station file
    pick_input      = Path(r"F:\SEML\DATA PICKING MEQ\DATA PICK 2023\PICK 2023 09\2023_09_full_test.xlsx")  # catalog picking
    calibration     = Path(r"F:\SEML\SEISMOMETER INSTRUMENT CORRECTION\CALIBRATION")                        # calibration file
    mw_result       = Path(r"F:\SEML\MAGNITUDE CALCULATION\Mw 2023\2023_09")                                # mw result location
    fig_out         = Path(r"F:\SEML\MAGNITUDE CALCULATION\Mw 2023\2023_09\fig_out")                        # spectral fitting figure location
   
    # dinamic input prompt
    id_start    = int(input("Event's ID to start the moment magnitude calculation: ")) 
    id_end      = int(input("Event's ID to end the moment magnitude calculation : ")) 
    mw_output   = str(input("Please input your desired moment magnitude calculation output name (ex. 2023_10): "))
    fig_status  = str(input("Do you want to produce the spectral fitting image [yes/no]?: "))
    
    # check status image builder
    if fig_status == 'yes':
        fig_state = True
    else:
        fig_state = False
    
    # loading file input
    hypo_data    = pd.read_excel(hypo_input, index_col = None) 
    pick_data    = pd.read_excel(pick_input, index_col = None)
    station_data = pd.read_excel(sta_input, index_col = None)

    # initiate dataframe for magnitude calculation results
    df_result   = pd.DataFrame(
                        columns = ["ID", "Fc", "Fc_std", "Mw", "Mw_std", "Src_rad(m)", "Src_rad_std", "Stress_drop(bar)", "Stress_drop_std"] 
                        )
    df_fitting  = pd.DataFrame(
                        columns = ["ID", "Station", "Component",  "F_corner_P", "F_corner_S", "Qfactor_P", "Qfactor_S", "Omega_0_P(nms)", "Omega_0_S(nms)", "RMS_e_P(nms)", "RMS_e_S(nms)"  ] 
                        )

    for _id in range (id_start, id_end + 1):
        
        print(f"Calculate moment magnitude for event ID {_id} ...")
        
        # get the dataframe 
        hypo_data_handler   = hypo_data[hypo_data["ID"] == _id]
        pick_data_handler   = pick_data[pick_data["Event ID"] == _id]

        # start calculating moment magnitude
        try:
            # calculate the moment magnitude
            mw_results, fitting_result = calculate_moment_magnitude(wave_path, hypo_data_handler, pick_data_handler, station_data, calibration, _id, fig_out, fig_state)

            # create the dataframe from calculate_ml_magnitude results
            mw_magnitude_result = pd.DataFrame.from_dict(mw_results)
            mw_fitting_result   = pd.DataFrame.from_dict(fitting_result)
            
            # concatinate the dataframe
            df_result = pd.concat([df_result, mw_magnitude_result], ignore_index = True)
            df_fitting = pd.concat([df_fitting, mw_fitting_result], ignore_index = True)
            
        except Exception as e:
            logger.exception("An error occured during runtime:", str(e))
            pass
            
    # save and set dataframe index
    df_result.to_excel(mw_result.joinpath(f"{mw_output}_result.xlsx"), index = False)
    df_fitting.to_excel(mw_result.joinpath(f"{mw_output}_fitting_result.xlsx"), index = False)
    print('-----------  The code has run succesfully! --------------')