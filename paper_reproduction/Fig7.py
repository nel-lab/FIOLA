#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:37:36 2021

@author: nel
"""

import caiman as cm
import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
#path = '/media/nel/DATA/fiola/sub-R2_ses-20190206T210000_obj-16d9pmi_behavior+ophys.nwb'
path = '/media/nel/DATA/fiola/R2_20190219/sub-R2_ses-20190219T210000_behavior+ophys.nwb'
io = NWBHDF5IO(path, 'r')
nwbfile_in = io.read()
mov = nwbfile_in.acquisition['TwoPhotonSeries'].data[:]
mov = cm.movie(mov)
mov.save('/media/nel/DATA/fiola/mov_R2_20190219T210000.hdf5')

#trace0 = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Deconvolved'].data[:]
trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:]
trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/fiola_30000.npy')
trace1 = trace1.T
tracem = np.load('/media/nel/DATA/fiola/R2_20190219/mean_roi_99.98.npy')
pos = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['pos'].data[:]
speed = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed'].data[:]

#%%
pos_n = pos.copy()
for i in range(len(pos_n)):
    if pos_n[i] < -2:
        j = i
        flag = 1
        while flag:
            j += 1
            try:
                if pos_n[j] >= -0.5:
                    pos_n[i] = pos_n[j]
                    flag = 0
            except:
                pos_n[i] = 0
                flag = 0
                
plt.figure()
plt.plot(pos)
plt.title('pos')
plt.plot(pos_n) 
plt.figure()
plt.plot(speed)
plt.title('spd')
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, Lasso
trace_s = StandardScaler().fit_transform(trace)
tracem_s = StandardScaler().fit_transform(tracem)
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]
trace1_s = StandardScaler().fit_transform(trace1)
trace1_s = trace1_s / trace1_s.max(0)

#%%
# from caiman.source_extraction.cnmf.temporal import constrained_foopsi
# dec = []
# for idx in range(len(trace1_s.T)):
#     c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(trace1_s[:, idx], p=1, s_min=0.1)
#     dec.append(s_full)
#     if idx in [25, 30, 35, 40, 45, 50]:
#         plt.title(f'neuron {idx}'); plt.plot(trace1_s[:, idx]); plt.plot(c_full-1); 
#         plt.plot(s_full-2, c='black'); plt.legend(['calcium', 'fit', 'dec']); plt.show()
# trace2 = np.array(dec).T
trace2 = trace1_s.copy()
trace2_s = StandardScaler().fit_transform(trace2)

#%%
from scipy.ndimage import gaussian_filter1d
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)
#%%
t1_g = trace2_s.copy()
for i in range(t1_g.shape[1]):
    t1_g[:, i] = gaussian_filter1d(t1_g[:, i], sigma=5)
    
#%%
tm_g = tracem_s.copy()
for i in range(tm_g.shape[1]):
    tm_g[:, i] = gaussian_filter1d(tm_g[:, i], sigma=5)
#%%
r = []
train = [0, 10000]
flag = 0
test = [10000, 12000]
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    #test = [10000, 12000]
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
    print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(r)}')
    
#%%
r1 = []
flag = 0
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(t1_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(t1_g[train[0]:train[1]])     
    print(f'test: {clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(t1_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    r1.append(clf.score(t1_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(r1)}')

#%%
rm = []
flag = 0
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.000001)
    clf.fit(tm_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(tm_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(tm_g[train[0]:train[1]])     
    print(f'test: {clf.score(tm_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(tm_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    rm.append(clf.score(tm_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        flag = 1
        plt.figure()
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(rm)}')

#%%
xx = list(range(10000, 30000, 2000))
plt.figure(); plt.plot(xx, r); plt.plot(xx, r1); plt.plot(xx, rm)
plt.plot(xx, np.array(r) - np.array(r1)); plt.legend(['Suite2p', 'Fiola', 'MeanROI', 'Difference Suite2p FIOLA'])
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar([0, 1, 2], [np.mean(r), np.mean(r1), np.mean(rm)], 
        yerr=[np.std(r), np.std(r1), np.std(rm)], label=['Suite2p', 'Fiola', 'MeanROI'])
ax1.legend()
#ax1.bar(0, np.mean(r), yerr=np.std(r))
#ax1.bar(0, np.mean(r1), yerr=np.std(r1))
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Average')
ax1.xaxis.set_ticks_position('none') 
#ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
#ax1.set_yticks([])
ax1.set_ylim([0,1])

print(np.mean(r))
print(np.mean(r1))
print(np.mean(rm))

#%%
r_all = []
for k_n in list(range(0, 2100, 200)):
    r_temp = []
    for it in range(5):
        trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:]
        idx = np.random.permutation(list(range(trace.shape[1])))[:k_n]
        trace[:, idx] = 1
        trace_s = StandardScaler().fit_transform(trace)
        
        t_g = trace_s.copy()
        for i in range(t_g.shape[1]):
            t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)
        
        r = []
        train = [0, 10000]
        flag = 1
        test = [10000, 12000]
        for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
            #test = [10000, 12000]
            clf = Ridge(alpha=800)
            #clf = LinearRegression()
            #clf = Lasso(alpha=0.000001)
            clf.fit(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
            pos_pre_test = clf.predict(t_g[test[0]:test[1]])     
            pos_pre_train = clf.predict(t_g[train[0]:train[1]])     
            print(f'test: {clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
            print(f'train: {clf.score(t_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
            r.append(clf.score(t_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
            if flag == 0:
                flag = 1
                plt.figure()
                plt.plot(pos_s[train[0]:train[1]])
                plt.plot(pos_pre_train)
                plt.figure()
                plt.plot(pos_s[test[0]:test[1]])
                plt.plot(pos_pre_test)
        print(f'average test: {np.mean(r)}')
        r_temp.append(np.mean(r))
    r_all.append(r_temp)
    
rr = [np.mean(r) for r in r_all]
plt.plot(list(range(50, 2250, 200)), rr[::-1])
plt.hlines(np.mean(r1), xmin=0, xmax=2000, color='orange')
plt.legend(['suite2p number of remaining', 'fiola 749 neurons with bg'])
plt.ylabel('R2')
plt.xlabel('number of neurons')

#%%
coef = clf.coef_.copy()
plt.plot(clf.coef_)
#idx = np.argsort(clf.coef_)[::-1][:40]
#idx = np.argsort(coef)[::-1][:300]
#idx1 = np.argsort(coef)[:300]
#idx = np.concatenate([idx, idx1])

trace.shape
idx = np.random.permutation(list(range(trace.shape[1])))[:1300]
trace[:, idx] = 1
plt.plot(t_g[:, idx])
plt.plot(pos_s)
plt.plot(trace_s[:, idx])
plt.plot(trace0[:, idx])


#%%
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g[train[0]:train[1]], spd_s[train[0]:train[1]])  
spd_pre_test = clf.predict(t_g[test[0]:test[1]])     
spd_pre_train = clf.predict(t_g[train[0]:train[1]])     
print(f'test: {clf.score(t_g[test[0]:test[1]], spd_s[test[0]:test[1]])}')
print(f'train: {clf.score(t_g[train[0]:train[1]], spd_s[train[0]:train[1]])}')
plt.figure()
plt.plot(spd_s[train[0]:train[1]])
plt.plot(spd_pre_train)
plt.figure()
plt.plot(spd_s[test[0]:test[1]])
plt.plot(spd_pre_test)




#%%
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t1_g[train[0]:train[1]], spd_s[train[0]:train[1]])  
spd_pre_test = clf.predict(t1_g[test[0]:test[1]])     
spd_pre_train = clf.predict(t1_g[train[0]:train[1]])     
print(f'test: {clf.score(t1_g[test[0]:test[1]], spd_s[test[0]:test[1]])}')
print(f'train: {clf.score(t1_g[train[0]:train[1]], spd_s[train[0]:train[1]])}')
plt.figure()
plt.plot(spd_s[train[0]:train[1]])
plt.plot(spd_pre_train)
plt.figure()
plt.plot(spd_s[test[0]:test[1]])
plt.plot(spd_pre_test)

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s, pos_s)  
pos_pre = clf.predict(trace_s)     
clf.score(trace_s, pos_s)

#%%
plt.figure()
plt.plot(pos_s) 
plt.plot(pos_pre, alpha=0.9)        
plt.title('pos')
plt.legend(['pos', 'fit'])

#%%
plt.plot(pos_n) 
plt.plot(spd_s)

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s, spd_s)  
spd_pre = clf.predict(trace_s)     
clf.score(trace_s, spd_s)

#%%
plt.figure()
plt.plot(spd_s) 
plt.plot(spd_pre, alpha=0.9)        
plt.title('spd')
plt.legend(['spd', 'fit'])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(trace_s[2500:12500], pos_s[2500:12500])  
pos_pre = clf.predict(trace_s[2500:12500])     
clf.score(trace_s[2500:12500], pos_s[2500:12500])

#%%
plt.plot(pos_s[2500:12500])
plt.plot(pos_pre)

#%%
from scipy.ndimage import gaussian_filter1d
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)

#%%
plt.plot(trace_s[2500:12500, 0])
plt.plot(t_g[:, 0])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g, pos_s[2500:12500])  
pos_pre = clf.predict(t_g)     
clf.score(t_g, pos_s[2500:12500])

#%%
clf = Ridge(alpha=1)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(t_g, pos_s)  
pos_pre = clf.predict(t_g)     
clf.score(t_g, pos_s)


#%%
plt.plot(pos_s)
plt.plot(pos_pre)
# #%%
# trace1_s = trace1.copy()
# for idx in range(len(trace1.T)):
#     t = trace1[:, idx]
#     window = 3000
#     weights = np.repeat(1.0, window)/window
#     t_pad = np.concatenate([t[:1500][::-1], t, t[-1499:]])
#     ma = np.convolve(t_pad, weights, 'valid')
#     # plt.plot(t)
#     # plt.plot(ma)
#     t = t - ma
#     trace1_s[:, idx] = t
#     if idx % 100 == 0:
#         print(idx)
        
# trace1_s = StandardScaler().fit_transform(trace1_s)


