#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:05:08 2020

@author: nel
"""

#%%
file_list = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/454597_Cell_0_40x_patch1_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A2_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_3_40x_1xtube_10A3_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A5_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A6_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/456462_Cell_5_40x_1xtube_10A7_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A1_output.npz',
             '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new/462149_Cell_1_40x_1xtube_10A2_output.npz']
fig = plt.figure(figsize=(12,12))
temp = file_list[0].split('/')[-1].split('_')
fig.suptitle(f'subject_id:{temp[0]}  Cell number:{temp[2]}')

pr= []
re = []
F = []
sub = []
N_opt = [2,2,2,2,2,2]
#new_Thr_opt = [8.5,8, 5.5, 10.5, 6.5, 10.5, 6.5, 6.5]
#new_F1 = [0.97,0.96, 0.82, 0.6, 0.39, 0.86, 0.67, 0.76]
#
#Thr_opt = [8.5,8.5, 5.5, 10.5, 6.5, 10.5, 6.5 ]
#F1 = [0.97, 0.96, 0.82, 0.6, 0.39, 0.86, 0.67]
#F1 = [0.98, 0.95, 0.69, 0.47, 0.88, 0.65, 0.95, 0.79]
all_f1_scores = []
all_corr_subthr = []

for file in file_list[1:2]:
    dict1 = np.load(file, allow_pickle=True)
    img = dict1['v_sg']
    print(np.diff( dict1['v_t']))
    std_estimate = np.diff(np.percentile(img,[75,25]))/4
    for i in range(len(dict1['sweep_time']) - 1):
        idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]), dict1['v_t']<dict1['sweep_time'][i+1][0])])[1]
        img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
    
    for i in range(len(dict1['sweep_time']) - 1):
        idx_to_rem = np.where([np.logical_and(dict1['v_t']>(dict1['sweep_time'][i][-1]-1), dict1['v_t']<dict1['sweep_time'][i][-1]-0.85)])[1]
        img[idx_to_rem] = np.random.normal(0,1,len(idx_to_rem))*std_estimate
        
    
    
    sub_1 = estimate_subthreshold(img, thres_STD=5)
    all_corr_subthr.append([np.corrcoef(normalize(dict1['e_sub']),normalize(sub_1))[0,1],np.corrcoef(normalize(dict1['e_sub']),normalize(dict1['v_sub']))[0,1]])
#    delta_img = np.diff(img)
    from caiman.source_extraction.volpy.spikepursuit import signal_filter
    sub_2 = signal_filter(dict1['v_sg'], 20, fr=400, order=5, mode='low') 

#%%
from caiman.source_extraction.volpy.spikepursuit import signal_filter
sub_2 = signal_filter(dict1['v_sg'], 20, fr=400, order=5, mode='low') 


#%%
def signal_filter(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg


#%%
    
from scipy import zeros, signal, random

data = random.random(2000)
data = dict1['v_sg']

def filter_sbs(data):
    fr=400
    freq=1
    normFreq = freq/(fr/2)
    b, a = signal.butter(3, normFreq, 'low')
    z = signal.lfilter_zi(b, a)
    result = zeros(data.size)
    for i, x in enumerate(data):
        result[i], z = signal.lfilter(b, a, [x], zi=z)
    return result

result = filter_sbs(data)
result_offline = signal_filter(dict1['v_sg'], 20, fr=400, order=5, mode='high')

plt.plot(data, label='orig')
#plt.plot(result_offline, label='offline')
plt.plot(result, label='online')
plt.legend()


#%%
def signal_filter_online(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg




