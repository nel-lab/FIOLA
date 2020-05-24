#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 08:12:26 2020

@author: agiovann
"""
import numpy as np
import pylab as plt
from running_statistics import estimate_running_std
#%%
def normalize(ss, remove_running=False):
    aa = (ss-np.min(ss))/(np.max(ss)-np.min(ss))
    aa -= np.median(aa)
    if remove_running:
        aa /= estimate_running_std(aa)
    return aa
  
#%%
def plot_poverlaid_results(img, dict1, dict1_v_sp_, erf=None, sub=None, v_match=None, e_match=None):
    eph = (dict1['e_sg']-np.mean(dict1['e_sg']))/(np.max(dict1['e_sg'])-np.min(dict1['e_sg']))
    if v_match is not None:
        FN = list(set(dict1_v_sp_)-set(v_match))
        FP = list(set(dict1['e_sp'])-set(e_match))
        plt.plot(FP,[1.06]*len(FP),'c|')
        plt.plot(FN,[1.04]*len(FN),'b|')
        
    plt.plot(dict1['e_sp'],[1.1]*len(dict1['e_sp']),'k.')
    plt.plot(dict1['v_sp'],[1.08]*len(dict1['v_sp']),'g.')

    plt.plot(dict1_v_sp_,[1.02]*len(dict1_v_sp_),'r.')
    plt.plot(dict1['v_t'][1:],normalize(np.diff(img)),'.-')
    plt.plot(dict1['e_t'],normalize(eph), color='k')
    plt.plot(dict1['v_t'],normalize(img),'-')
    plt.plot(dict1['v_t'][1:],-erf/np.max(-erf),'r-')
    if sub is not None:
        plt.plot(dict1['v_t'],normalize(subs))
#%%
def plot_marton(e_spike_aligned, v_spike_aligned, e_match, v_match, mean_time, precision, recall, F1, sub_corr):
        fig = plt.figure()
        ax1 = fig.add_axes([0.05, 0.8, 0.9, 0.15])
        e_fr = np.unique(np.floor(e_spike_aligned), return_counts=True)
        v_fr = np.unique(np.floor(v_spike_aligned), return_counts=True)
        ax1.plot(e_fr[0], e_fr[1], color='black')
        ax1.plot(v_fr[0], v_fr[1], color='g')
        ax1.legend(['ephys','voltage'])
        ax1.set_ylabel('Firing Rate (Hz)')
        
        
        ax2 = fig.add_axes([0.05, 0.6, 0.9, 0.15])
        ax2.vlines(list(set(v_spike_aligned)-set(v_match)), 2.75,3.25, color='red')
        ax2.vlines(v_spike_aligned, 1.75,2.25, color='green')
        ax2.vlines(e_spike_aligned, 0.75,1.25, color='black')
        ax2.vlines(list(set(e_spike_aligned)-set(e_match)), -0.25,0.25, color='red')
        plt.yticks(np.arange(4), ['False Negative', 'Ephys', 'Voltage', 'False Positive'])
        
        ax3 = fig.add_axes([0.05, 0.2, 0.9, 0.35])
        ax3.plot(mean_time, precision, 'o-', c='blue')
        ax3.plot(mean_time, recall, 'o-', c='orange')
        ax3.plot(mean_time, F1, 'o-', c='green')
        
        
        ax4 = fig.add_axes([0.05, 0, 0.9, 0.15])
        ax4.plot(mean_time, sub_corr, 'o-', c='blue')
    