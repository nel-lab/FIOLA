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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, Lasso
#path = '/media/nel/DATA/fiola/sub-R2_ses-20190206T210000_obj-16d9pmi_behavior+ophys.nwb'
path = '/media/nel/DATA/fiola/R2_20190219/sub-R2_ses-20190219T210000_behavior+ophys.nwb'
#path = '/media/nel/DATA/fiola/R4_20190904/sub-R4_ses-20190904T210000_behavior+ophys.nwb'
#path = '/media/nel/DATA/fiola/F2_20190415/sub-F2_ses-20190415T210000_behavior+ophys.nwb'
io = NWBHDF5IO(path, 'r')
nwbfile_in = io.read()
pos = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['pos'].data[:]
speed = nwbfile_in.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series['speed'].data[:]

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
                
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]
                
# plt.figure(); 
# plt.plot(pos)
# plt.title('pos')
# plt.plot(pos_n) 
# plt.figure()
# plt.plot(speed)
# plt.title('spd')
# 
# mov_100 = nwbfile_in.acquisition['TwoPhotonSeries'].data[:1000]
# #m1 = mov_100.mo
# mov_l_100 = nwbfile_in.acquisition['TwoPhotonSeries'].data[-5000:-4000]
# me = np.median(mov_100, axis=0)
# me1 = np.median(mov_l_100, axis=0)
# mm = cm.concatenate([cm.movie(me)[None, :, :], cm.movie(me1)[None, :, :]])
# mov = cm.movie(mov)
#mov.save('/media/nel/DATA/fiola/mov_R2_20190219T210000.hdf5')
# mov[:1000].save('/media/nel/DATA/fiola/R2_20190219/mov_R2_20190219T210000_1000.hdf5')
# mov[:5000].save('/media/nel/DATA/fiola/R2_20190219/mov_R2_20190219T210000_5000.hdf5')
# mov[:7000].save('/media/nel/DATA/fiola/R2_20190219/mov_R2_20190219T210000_7000.hdf5')

# mov.save('/media/nel/DATA/fiola/mov_R2_20190219T210000.hdf5')

# masks = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[0][:]
# centers = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[1][:]
# #accepted = nwbfile_in.processing['ophys'].data_interfaces['ImageSegmentation'].plane_segmentations['PlaneSegmentation'].columns[2][:]
# plt.imshow(masks.sum(0))
# plt.scatter(centers[:, 0], centers[:, 1],  s=0.5, color='red')

#%%
trace0 = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['Deconvolved'].data[:]
trace = nwbfile_in.processing['ophys'].data_interfaces['Fluorescence'].roi_response_series['RoiResponseSeries'].data[:]
#trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/3000/fiola_30000.npy')
#trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/3000/fiola_3000_nonrigid.npy')
#trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/3000/fiola_30000_K_8.npy')
#trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/3000/fiola_1285_nonrigid.npy')
trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie.npy')
trace1 = np.load('/media/nel/DATA/fiola/R2_20190219/3000/fiola_1285_non_rigid_init_non_rigid_movie_nnls_only.npy')
trace1 = trace1.T
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/3000/mean_roi_99.98.npy')
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_all_masks.npy')
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/3000/mean_roi_99.98_non_rigid.npy')
#tracem = tracem.T
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks.npy')
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_1543.npy')
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_2365.npy')
#tracem = np.load('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/mean_roi_99.98_non_rigid_caiman_3000_masks_selected_1285.npy')
tracem = np.load('/media/nel/DATA/fiola/R2_20190219/3000/mean_roi_99.98_non_rigid_init_non_rigid_1285.npy')
#%%
from sklearn.preprocessing import StandardScaler
from caiman.source_extraction.cnmf.cnmf import load_CNMF
cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_K_8.hdf5')
#cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/caiman_online_results.hdf5')
#cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8_cnn_True.hdf5')
#cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5_snr_1.8.hdf5')
#cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp_5_5.hdf5')
#cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full_nonrigid/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_all_comp.hdf5')

#cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/full/memmap__d1_796_d2_512_d3_1_order_C_frames_31933_.hdf5')
tracec = cnm2.estimates.C[cnm2.estimates.idx_components] + cnm2.estimates.YrA[cnm2.estimates.idx_components]
#tracec = cnm2.estimates.C + cnm2.estimates.YrA
tracec = tracec.T
#tracec = tracec[1:]
#tracec = np.vstack([np.zeros((1, 1630)), tracec])
#tracec_s = StandardScaler().fit_transform(tracec)
#%%
trace1_s = StandardScaler().fit_transform(trace1)
tracem_s = StandardScaler().fit_transform(tracem)


#%%
def remove_neurons_with_sparse_activities(trace, do_remove=True, std_level=5, timepoints=10):
    from sklearn.preprocessing import StandardScaler
    print(f'original trace shape {trace.shape}')
    trace_s = StandardScaler().fit_transform(trace)
    if do_remove:
        select = []
        high_act = []
        for idx in range(len(trace_s.T)):
            t = trace_s[:, idx]
            select.append(len(t[t>t.std() * std_level]) > timepoints)
            high_act.append(len(t[t>t.std() * std_level]))
        #plt.plot(trace_s[:, ~np.array(select)][:, 15])
        trace_s = trace_s[:, select]
        high_act = np.array(high_act)[select]
        sort = np.argsort(high_act)[::-1]
        trace_s = trace_s[:, sort]
    print(f'after removing neurons trace shape {trace_s.shape}')
    return trace_s, select
   
trace_s = remove_neurons_with_sparse_activities(trace)
trace1_s = remove_neurons_with_sparse_activities(trace1)
tracem_s = remove_neurons_with_sparse_activities(tracem)
tracec_s = remove_neurons_with_sparse_activities(tracec)

#%%
first_act = []
std_level = 5
for idx in range(len(tracec_s.T)):
    t = tracec_s[:, idx]
    first_act.append(np.where(t>t.std() * std_level)[0][0])
    
plt.plot(first_act)
select = np.where(np.array(first_act)<3000)[0]
tracec_s = tracec_s[:, select]
#%%
from scipy.ndimage import gaussian_filter1d
t_g = trace_s.copy()
for i in range(t_g.shape[1]):
    t_g[:, i] = gaussian_filter1d(t_g[:, i], sigma=5)

t1_g = trace1_s.copy()
for i in range(t1_g.shape[1]):
    t1_g[:, i] = gaussian_filter1d(t1_g[:, i], sigma=5)
    
tm_g = tracem_s.copy()
for i in range(tm_g.shape[1]):
    tm_g[:, i] = gaussian_filter1d(tm_g[:, i], sigma=5)
   
#%%
from scipy.ndimage import gaussian_filter1d
tc_g = tracec_s.copy()
for i in range(tc_g.shape[1]):
    tc_g[:, i] = gaussian_filter1d(tc_g[:, i], sigma=5)

#%%
import numpy as np
from sklearn.model_selection import KFold
def cross_validation_ridge(X, y, n_splits=5, alpha=500):
    score = []
    kf = KFold(n_splits=5)
    for cv_train, cv_test in kf.split(X):
        x_train = X[cv_train]
        y_train = y[cv_train]
        x_test = X[cv_test]
        y_test = y[cv_test]
    
        clf = Ridge(alpha=alpha)
        clf.fit(x_train, y_train)  
        score.append(clf.score(x_test, y_test))
    return score

#X = t_g[:30000]
method = ['Suite2p', 'Fiola', 'MeanROI', 'CaImAn'][1:3]
#X_list = [t_g, t1_g, tm_g, tc_g][1:2]
X_list = [t1_g, tm_g]

r = {}
alpha_list = [100, 500, 1000, 5000, 10000]
for idx, X in enumerate(X_list):
    print(method[idx])
    X = X[:30000]
    y = pos_s[:30000]
    score_all = {}
    for alpha in alpha_list:
        print(alpha)
        score = cross_validation_ridge(X, y, n_splits=5, alpha=alpha)
        score_all[alpha] = score
    #print(score_all)
    score_m = [np.mean(s) for s in score_all.values()]
    print(f'max score:{max(score_m)}')
    print(f'alpha:{alpha_list[np.argmax(score_m)]}')
    r[method[idx]] = score_all[alpha_list[np.argmax(score_m)]]
    
#%%
for key in r.keys():
    print(np.mean(r[key]))

#%%
def rand_jitter(arr):
    stdev = .01 #* (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)
for i in range(4):
    jitter([i]*5, r[method[i]])
    
plt.ylim([0, 1])
plt.xticks(list(range(4)), method)
plt.title('R^2 for prediction with cross validation')

#plt.scatter([1, 1], [1, 2])
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
        plt.title(f'{test}')
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
        #flag = 1
        plt.figure()
        plt.title(f'{test}')
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
rc = []
flag = 0
train = [0, 10000]
for test in [[i, i+2000] for i in range(10000, 30000, 2000)]:
    clf = Ridge(alpha=800)    
    #clf = LinearRegression()
    #clf = Lasso(alpha=0.05)
    clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
    pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
    pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
    print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
    print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
    rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
    if flag == 0:
        #flag = 1
        plt.figure()
        plt.title(f'{test}')
        plt.plot(pos_s[train[0]:train[1]])
        plt.plot(pos_pre_train)
        plt.figure()
        plt.plot(pos_s[test[0]:test[1]])
        plt.plot(pos_pre_test)
print(f'average test: {np.mean(rc)}')

#%%
rc = []
flag = 0
sets = [[0, 10000], [10000, 20000], [20000, 30000]]
for train in sets:
    for test in [[i, i+10000] for i in range(0, 30000, 10000)]:
        clf = Ridge(alpha=800)    
        #clf = LinearRegression()
        #clf = Lasso(alpha=0.05)
        clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
        pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
        pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
        print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
        print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
        rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
        if flag == 0:
            #flag = 1
            plt.figure()
            plt.title(f'{test}')
            plt.plot(pos_s[train[0]:train[1]])
            plt.plot(pos_pre_train)
            plt.figure()
            plt.plot(pos_s[test[0]:test[1]])
            plt.plot(pos_pre_test)
    print(f'average test: {np.mean(rc)}')
#%%
xx = list(range(10000, 30000, 2000))
plt.figure(); plt.plot(xx, r); plt.plot(xx, r1); plt.plot(xx, rm); plt.plot(xx, rc)
plt.plot(xx, np.array(r) - np.array(r1)); 
plt.legend(['Suite2p', 'Fiola', 'MeanROI','CaImAn', 'Difference Suite2p FIOLA'])

#%%
fig = plt.figure(figsize=(8, 6)) 
ax1 = plt.subplot()
ax1.bar([0, 1, 2, 3], [np.mean(r), np.mean(r1), np.mean(rm), np.mean(rc)], 
        yerr=[np.std(r), np.std(r1), np.std(rm), np.std(rc)], label=['Suite2p', 'Fiola', 'MeanROI', 'CaImAn'])
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
print(np.mean(rc))

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
corr = []
for idx in range(len(tracem.T)):
    corr.append(np.corrcoef(tracem_s[:-1, idx], pos_n)[0, 1])



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

#%%
dims = (796, 512)
li = []
li_matrix = np.zeros((len(range(0, dims[0], 30)), len(range(0, dims[1], 30))))
for i in range(0, dims[0], 30):
    for j in range(0, dims[1], 30):
        num = 0
        for idx in range(centers.shape[0]):
            if centers[idx, 1] >= i and centers[idx, 1] < i + 30:
                if centers[idx, 0] >= j and centers[idx, 0] < j + 30:
                   num = num + 1
        li.append(num)
        li_matrix[i//30, j//30] = num
#li_matrix[li_matrix<] = 0
plt.imshow(li_matrix); plt.colorbar
#plt.hist(li_matrix.flatten())#; plt.colorbar()

#%%
from caiman.base.rois import com
centers_cm = com(cnm2.estimates.A[:, cnm2.estimates.idx_components], 796, 512)
centers_cm = centers_cm[:, np.array([1, 0])]
centers = centers_cm.copy()
#%%
dims = (796, 512)
A = cnm2.estimates.A.toarray()
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])

aa = np.load('/media/nel/DATA/fiola/R2_20190219/3000/caiman_masks_3000_749.npy', allow_pickle=True)

#%%
cnm2 = load_CNMF('/media/nel/DATA/fiola/R2_20190219/7000/memmap__d1_796_d2_512_d3_1_order_C_frames_7000_.hdf5')
A = cnm2.estimates.A.toarray()
A = A.reshape([dims[0], dims[1], -1], order='F')
A = A.transpose([2, 0, 1])

#%%
r = []
train = [0, 10000]
flag = 1
test = [10000, 30000]
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
    #flag = 1
    # plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    plt.figure()
    plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(r)}')

r1 = []
flag = 1
#test = [10000, 12000]
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
    #flag = 1
    #plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    #plt.figure()
    #plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(r1)}')


rc = []
flag = 1
#test = [10000, 12000]
clf = Ridge(alpha=800)
#clf = LinearRegression()
#clf = Lasso(alpha=0.000001)
clf.fit(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])  
pos_pre_test = clf.predict(tc_g[test[0]:test[1]])     
pos_pre_train = clf.predict(tc_g[train[0]:train[1]])     
print(f'test: {clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]])}')
print(f'train: {clf.score(tc_g[train[0]:train[1]], pos_s[train[0]:train[1]])}')
rc.append(clf.score(tc_g[test[0]:test[1]], pos_s[test[0]:test[1]]))
if flag == 0:
    #flag = 1
    #plt.figure()
    # plt.plot(pos_s[train[0]:train[1]])
    # plt.plot(pos_pre_train)
    #plt.figure()
    #plt.plot(pos_s[test[0]:test[1]])
    plt.plot(pos_pre_test)
print(f'average test: {np.mean(rc)}')

#%%

trace_s = StandardScaler().fit_transform(trace)
tracem_s = StandardScaler().fit_transform(tracem)
pos_s = StandardScaler().fit_transform(pos_n[:, None])[:, 0]
spd_s = StandardScaler().fit_transform(speed[:, None])[:, 0]
trace1_s = StandardScaler().fit_transform(trace1)
trace1_s = trace1_s / trace1_s.max(0)

#%%
tracec_s = StandardScaler().fit_transform(tracec)

#%%
#tracem_s = tracem_s / tracem_s.max(0)
# tracem_s= tracem_s - tracem_s.min(0)
# tracem_s = tracem_s / tracem_s.max(0)
# tracem_s = tracem_s/tracem_s.max(0)
# for i in range(tracem_s.shape[1]):
#     tracem_s[:, i] = gaussian_filter1d(tracem_s[:, i], sigma=5)
ii = 101
plt.plot(trace_s[:, ii])
print((trace_s[:3000, ii] > 5).sum())
#%%
n = []
for ii in range(trace_s.shape[1]):
    n.append((trace_s[:3000, ii] > 5).sum())
    
n = np.array(n)
(n > 5).sum()
select = np.where(n>5)[0]
#%%
# from caiman.source_extraction.cnmf.temporal import constrained_foopsi
# dec = []
# for idx in range(len(tracem_s.T)):
#     c_full, bl, c1, g, sn, s_full, lam = constrained_foopsi(tracem_s[:, idx], p=1, s_min=0.1)
#     dec.append(s_full)
#     if idx in [125, 130, 135, 140, 145, 150]:
#         plt.title(f'neuron {idx}'); plt.plot(tracem_s[:, idx]); plt.plot(c_full-3); 
#         plt.plot(s_full-6, c='black'); plt.legend(['calcium', 'fit', 'dec']); plt.show()
# trace2 = np.array(dec).T

# trace2 = np.array(dec).T
trace2 = trace1_s.copy()
trace2_s = StandardScaler().fit_transform(trace2)
t1_g = trace2_s.copy()

from caiman.source_extraction.volpy.spikepursuit import signal_filter
tt = signal_filter(trace1, freq=1, fr=15)

# plt.plot(trace1[:, 30]); plt.plot(tt[:, 30])
plt.plot(trace1[:, :].sum(1)); plt.plot(tt[:, :].sum(1))
trace1 = tt
#trace1 = trace1[1:]