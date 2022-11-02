#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:02:16 2020
This file is for timing of spike detection algorithm
@author: caichangjia
"""

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})
import numpy as np
import pyximport
pyximport.install()
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter



#%%
frate=400
saoz_all = {}
for num_neurons in [100]:#, 500]:
    img = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/test_timing.npy')
    img = np.hstack([img[:20000] for _ in range(5)])
    trace_all = np.stack([img for i in range(num_neurons)])
    saoz = SignalAnalysisOnlineZ(fr=frate, flip=False, robust_std=False, do_scale=False, detrend=False)
    saoz.fit(trace_all[:,:20000], len(img))
    # for n in range(20000, trace_all.shape[1]):
    #     saoz.fit_next(trace_all[:, n:n+1], n)
    for n in range(20000, 100000):
        saoz.fit_next(trace_all[:, n:n+1], n)
    saoz_all[num_neurons] = saoz

#%%
plt.figure()
#plt.plot(saoz.t_detect)
plt.plot(np.array(saoz_all[500].t_detect)[20000:]*1000, label='500 neurons', color='orange')
plt.plot(np.array(saoz_all[100].t_detect)[20000:]*1000, label='100 neurons', color='blue')
plt.ylabel('Timing (ms)')
plt.legend()
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_v3.7.pdf')

#%%
t1 = np.array(saoz_all[100].t_detect[20000:]) * 1000
t2 = np.array(saoz_all[200].t_detect[20000:]) * 1000
t3 = np.array(saoz_all[500].t_detect[20000:]) * 1000

plt.figure()
plt.bar([0, 1, 2], [t1.mean(), t2.mean(), t3.mean()], yerr=[t1.std(), t2.std(), t3.std()])
plt.ylabel('Timing (ms)')
#plt.xlabel('Number of neurons')
plt.xticks([0, 1, 2], ['100 neurons', '200 neurons', '500 neurons'])
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_mean_v3.7.pdf.pdf')


#%%
"""
from nmf_support import normalize
plt.plot(dict1['v_t'], saoz.t_detect)
plt.plot(dict1['e_t'], dict1['e_sg']/100000)

