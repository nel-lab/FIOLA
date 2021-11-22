#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:02:16 2020
This file is for timing of spike detection algorithm
@author: caichangjia
"""

#%%
from fiola.utilities import signal_filter
import matplotlib.pyplot as plt
import numpy as np
from fiola.signal_analysis_online import SignalAnalysisOnlineZ

#%%
frate=400
img = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/test_timing.npy')
trace_all = np.stack([img for i in range(100)])
saoz = SignalAnalysisOnlineZ(fr=frate, robust_std=False, do_scale=True)
saoz.fit(trace_all[:,:20000], len(img))
for n in range(20000, trace_all.shape[1]):
    saoz.fit_next(trace_all[:, n:n+1], n)

#%%
plt.figure()
#plt.plot(saoz.t_detect)
plt.plot(saoz1.t_detect, label='500 neurons')
plt.plot(saoz2.t_detect, label='100 neurons')
plt.legend()
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/timing/timing_spikes_detection_50neurons.pdf')

#%%
"""
from nmf_support import normalize
plt.plot(dict1['v_t'], saoz.t_detect)
plt.plot(dict1['e_t'], dict1['e_sg']/100000)

