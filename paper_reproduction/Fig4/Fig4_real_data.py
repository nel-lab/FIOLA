#!/usr/bin/env python
#%%
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os

from paper_reproduction.Fig6.Fig6_utilities import barplot_annotate_brackets, barplot_pvalue
from paper_reproduction.utilities import multiple_dfs


#%%
import logging
logging.basicConfig(level = logging.DEBUG)
    
#%% Fig 4e
names = ['454597_Cell_0_40x_patch1', '456462_Cell_3_40x_1xtube_10A2',
         '456462_Cell_3_40x_1xtube_10A3', '456462_Cell_5_40x_1xtube_10A5',
         '456462_Cell_5_40x_1xtube_10A6', '456462_Cell_5_40x_1xtube_10A7', 
         '462149_Cell_1_40x_1xtube_10A1', '462149_Cell_1_40x_1xtube_10A2',
         '456462_Cell_4_40x_1xtube_10A4', '456462_Cell_6_40x_1xtube_10A10',
         '456462_Cell_5_40x_1xtube_10A8', '456462_Cell_5_40x_1xtube_10A9', 
         '462149_Cell_3_40x_1xtube_10A3', '466769_Cell_2_40x_1xtube_10A_6',
         '466769_Cell_2_40x_1xtube_10A_4', '466769_Cell_3_40x_1xtube_10A_8', 
         '09282017Fish1-1', '10052017Fish2-2', 'Mouse_Session_1']
#save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
labels = names
#lists1 = [li[:13]   for li in lists]
#labels = lists1
x = np.arange(len(labels))  # the label locations
width = 0.66  # the width of the bars

result = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500.npy', allow_pickle=True).item()
viola = result['viola']['f1']
volpy = result['volpy']['f1']
viola1 = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500_filt_window_[8, 4]_template_window_2.npy', allow_pickle=True)
viola1 = viola1.item()['viola']['f1']
viola2 = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500_filt_window_[8, 4]_template_window_0.npy', allow_pickle=True)
viola2 = viola2.item()['viola']['f1']
meanroi = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/mean_roi_threshold_2.9.npy', allow_pickle=True)
meanroi = meanroi.item()['f1']
v = np.array([viola, volpy, viola1, viola2, meanroi])
v_mean = v.mean(1)
v_std = v.std(1)


#%%
amps = [0]
spk_amps = [0]
colors = ['C0', 'C1', 'C2', 'C3']
methods = ['FIOLA', 'VolPy', 'FIOLA20', 'FIOLA15']

r_all = []
rr = {}
batches = [0]
n_batch = len(batches)

for batch in batches:
    for jj, method in enumerate(methods):
        results = v[jj]
        rr[method] = results
    r_all.append(rr)
    rr = {}

fig = plt.figure(figsize=(8, 10))
ax = plt.subplot(111)
p_value = barplot_pvalue(r_all, methods, colors, ax, width=0.3, dev=0.003, capsize=8)

ax.set_xlabel('')
ax.set_ylabel('F1 score')
#ax.set_ylim([0, 1.2])
ax.set_xticks(amps)
ax.set_xticklabels(spk_amps)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.tight_layout()
#plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/Fig4/Fig4h_v3.10.pdf')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'
labels = ['FIOLA', 'VolPy', 'FIOLA20', 'FIOLA15']
#results = result_all.copy()

df1 = pd.DataFrame({})
for idx1, s1 in enumerate(labels):
    rr = r_all[0][s1]
    df1[s1] = rr
        
# df2 = pd.DataFrame({'spike amplitude':x})
# for idx1 in range(len(labels)):
#     df2[labels[idx1]] = data_points[idx1]

dfs = [df1]
text = 'Average F1 score against ephys ground truth'
fig_name = 'Fig 4e'
excel_name = os.path.join(excel_folder, 'Fig4.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/p_value'
labels = methods
results = r_all.copy()

df1 = pd.DataFrame({})
for mmm in p_value:
    df1[mmm] = p_value[mmm]
dfs = [df1]
fig_name = 'Sheet'
excel_name = os.path.join(excel_folder, f'Fig 4e.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 0, text=None, row=0)

#%% Fig4f
fig, ax = plt.subplots(1, 1)
xx = np.arange(0.2, 1.1, 0.01)
yy = xx.copy()
ax.plot(xx, yy, '--', color='black')
ax.scatter(v[0], v[1], color='black')
ax.set_xlabel('FIOLA'); ax.set_ylabel('VolPy'); 
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(length=8)
ax.yaxis.set_tick_params(length=8)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim([0.15, 1.05])
ax.set_xlim([0.15, 1.05])
plt.gca().set_aspect('equal', adjustable='box')
#save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
#plt.savefig(os.path.join(save_folder, 'one_neuron_f1_score.pdf'))


#%% Fig4g
from sklearn.linear_model import LinearRegression
x_test = np.arange(1, 7.1, 0.1)[:, np.newaxis]
x = [round(0.05 + 0.025 * i, 3) for i in range(7)]    
VIOLA_FOLDER = '/media/nel/storage//NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' in file]
files = viola_files
result_all = [np.load(file, allow_pickle=True) for file in files][0]
plt.plot(x, [np.array(result).sum()/len(result) for result in result_all],  marker='.', markersize=15)
spnr_sim = np.array([np.array(result).sum()/len(result) for result in result_all])

VIOLA_FOLDER = '/media/nel/storage//NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_simulations/non_overlapping/viola_result_online_gpu_True_init_10000_bg_True_use_spikes_False_hals_movie_hp_thresh_semi_nmf_False_adaptive_threshold_True_do_scale_False_freq_15_v2.0'
viola_files = os.listdir(VIOLA_FOLDER)
viola_files = [os.path.join(VIOLA_FOLDER, file) for file in viola_files if 'v2.0.npy' in file and 'spnr' not in file]
files = viola_files
result_all = [np.load(file, allow_pickle=True) for file in files][0]
# plt.plot(x, [np.array(result['F1']).sum()/len(result['F1']) for result in result_all], marker='.', markersize=15)    
# plt.plot(spnr_sim, F1_sim, marker='.', markersize=15, color='orange')    

F1_sim = np.array([np.array(result['F1']).sum()/len(result['F1']) for result in result_all])

spnr_real = np.array(np.load('/media/nel/storage//NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500_filt_window_15_template_window_2.npy', allow_pickle=True).item()['viola']['spnr'])
result = np.load('/media/nel/storage//NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500.npy', allow_pickle=True).item()
F1_real = np.array(result['viola']['f1']).copy()

#%%
fig, ax = plt.subplots(1, 1)
ax.scatter(spnr_real, F1_real, label='real data'); ax.set_xlabel('SPNR'); ax.set_ylabel('F1 score')
ax.scatter(spnr_sim, F1_sim, color='orange', label='simulation')    
#ax.plot(x_test.flatten(), y_real, label='real data')
#ax.plot(x_test.flatten(), y_real_ransac, label='real data ransac')
#ax.plot(x_test.flatten(), y_real_huber, label='real data huber')
#ax.plot(x_test.flatten(), y_real_theilsen, label='real data theilsen')
#ax.plot(x_test.flatten(), y_sim, label='simulation')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_tick_params(length=8)
ax.yaxis.set_tick_params(length=8)
ax.set_ylim([0.1, 1])
ax.set_xticks([2, 4, 6, 8])
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
#plt.savefig(os.path.join(save_folder, 'one_neuron_spnr_vs_f1_score.pdf'))

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'
labels = ['Real data', 'Simulation']
results = result_all.copy()

df1 = pd.DataFrame({})
df1['Real data' + '_' + 'F1 score'] = F1_real
df1['Real data' + '_' + 'SPNR'] = spnr_real

df2 = pd.DataFrame({})
df2['Simulation' + '_' + 'F1 score'] =  F1_sim
df2['Simulation' + '_' + 'SPNR'] = spnr_sim


dfs = [df1, df2]
text = 'F1 score vs SPNR for real data and simulation'
fig_name = 'Fig 4g'
excel_name = os.path.join(excel_folder, 'Fig4.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Fig 4j, Supp fig 5b, supp fig 6d
result_all = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neuron/viola_F1_v2.1.npy', 
                     allow_pickle=True).item()
F1 = np.zeros([3,8])

for keys, values in result_all.items():
    print(keys)
    if '0&1' in keys:
        row = 0        
    if '0&2' in keys:
        row = 1
    if '1&2' in keys:
        row = 2
    print(row)
    
    if '_25percent' in keys:
        col = 6
    if '_10percent' in keys:
        col = 4
    if '_0percent' in keys:
        col = 2
    print(col)
        
    for kk, vv in values.items():
        F1[row, col] = vv['f1']
        col = col + 1
        
result = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500.npy',
                 allow_pickle=True).item()
rr = result['viola']['f1'].copy()
F1[0,0] = rr[0]
F1[1,0] = rr[0]
F1[0,1] = rr[1]
F1[1,1] = rr[2]
F1[2,0] = rr[1]
F1[2,1] = rr[2]


#%% Fig 4j 
save_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/'
label = ['Cell1', 'Cell2']
#x = np.arange(len(label))  # the label locations
x = np.array([0, 1])
width = 0.1  # the width of the bars

ff = F1[0].reshape((2,4), order='F')
colors=['blue', 'yellow', 'red', 'green']
methods = ['seperate', '0%', '10%', '25%']

fig = plt.figure()
ax = plt.subplot(111)
#ax1 = plt.subplot(122)

for idx in range(4):
    ax.bar(x+width*(idx-1.5), ff[:, idx], width, color=colors[idx], label=methods[idx])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_tick_params(length=8)
ax.set_ylim([0.5,1])
ax.legend(frameon=False,  loc='upper center')

#%%
excel_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/FIOLA_nature_methods_final_submission/source_data'
labels = ['Cell1', 'Cell2']

df1 = pd.DataFrame({})
df1['methods'] = methods
for idx1, s1 in enumerate(labels):
    df1[s1] = ff[idx1]
        
dfs = [df1]
text = 'F1 score with different overlapping area'
fig_name = 'Fig 4j'
excel_name = os.path.join(excel_folder, 'Fig4.xlsx')# run function
multiple_dfs(dfs, fig_name, excel_name, 2, text)

#%% Supp fig 5b
save_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/'

#x = np.arange(len(label))  # the label locations
x = np.array([0, 1])
width = 0.1  # the width of the bars
colors=['blue', 'yellow', 'red', 'green']
methods = ['seperate', '0%', '10%', '25%']

fig, ax = plt.subplots(1, 3, figsize=(15, 8))
for j in range(3):
    if j == 0:
        label = ['Cell1', 'Cell2']
    elif j == 1:
        label = ['Cell1', 'Cell3']
    elif j == 2:
        label = ['Cell2', 'Cell3']
    
    ff = F1[j].reshape((2,4), order='F')
    for idx in range(4):
        ax[j].bar(x+width*(idx-1.5), ff[:, idx], width, color=colors[idx], label=methods[idx])
    if j == 0:
        ax[j].set_ylabel('F1 Score')
        ax[j].legend(frameon=False,  loc='upper center')
    ax[j].set_xticks(x)
    ax[j].set_xticklabels(label)
    ax[j].xaxis.set_ticks_position('none') 
    ax[j].yaxis.set_tick_params(length=8)
    ax[j].set_ylim([0.5,1])
    
plt.tight_layout()
#plt.savefig(save_folder+'v3.0/supp/Fig_supp_real_data_overlapping.pdf')


#%% Supp fig 6d
colors=['blue', 'orange']
methods = ['FIOLA', 'meanroi online']

fig = plt.figure()
ax = plt.subplot(111)

x = np.array([0, 1, 2 ,3])
label = ['seperate', '0%', '10%', '25%']
width = 0.2
for idx in range(2):
    ax.bar(x+width*(idx-0.5), [ff[0], ff1[0]][idx], width, color=colors[idx], label=methods[idx])

ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.set_ylim([0.3,1])
ax.legend()
fig.tight_layout()
#plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_real_data_ablation.pdf')

#%% Supp fig 4a
Cn = mov[0]
vmax = np.percentile(Cn, 99)
vmin = np.percentile(Cn, 5)
plt.figure()
plt.imshow(Cn, interpolation='None', vmax=vmax, vmin=vmin, cmap=plt.cm.gray)
plt.axis('off')
#plt.title('Neurons location')
d1, d2 = Cn.shape
#cm1 = com(mask.copy().reshape((N,-1), order='F').transpose(), d1, d2)
colors='yellow'
from skimage import measure
for n, idx in enumerate(idx_list):
    contours = measure.find_contours(mask0[idx], 0.5)[0]
    plt.plot(contours[:, 1], contours[:, 0], linewidth=1, color=colorsets[np.mod(n,9)])

save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1/Fig6'
#plt.savefig(os.path.join(save_folder, name + '_spatial.pdf'))

#%% Supp fig 4b
from paper_reproduction.Fig7.Fig7_utilities import *

folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_multiple_neurons'
files = sorted(os.listdir(folder))
viola = []
viola_std = []
volpy = []
volpy_std = []

labels = ['L1', 'zebra', 'HPC']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars    
from matplotlib import gridspec
fig = plt.figure(figsize=(8, 6)) 
#gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1]) 
ax0 = plt.subplot()
#ax1 = plt.subplot(gs[1])

for i, file in enumerate(files):
    mm = np.load(os.path.join(folder, file), allow_pickle=True)
    mm = mm[~np.isnan(mm[:, 0])]
    print(mm)
    spnr = np.nanmean(mm, axis=0)
    std = np.nanstd(mm, axis=0)
    viola.append(spnr[0])
    volpy.append(spnr[1])
    viola_std.append(std[0])
    volpy_std.append(std[1])
    
    ax0.scatter(rand_jitter([i+1 - width/2]*mm.shape[0], dev=0.01), mm[:, 0], facecolor='none', alpha=1, s=15, zorder=2, color='black')
    ax0.scatter(rand_jitter([i+1 + width/2]*mm.shape[0], dev=0.01), mm[:, 1], facecolor='none', alpha=1, s=15, zorder=2, color='black')

rects1 = ax0.bar(x+1 - width/2, viola, width, yerr=viola_std, label=f'Fiola', capsize=5)
rects2 = ax0.bar(x+1 + width/2, volpy, width, yerr=volpy_std, label=f'VolPy', capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_xlabel(names)
ax0.set_xticks(x+1)
ax0.set_ylabel('SpNR')
ax0.xaxis.set_ticks_position('none') 
ax0.yaxis.set_tick_params(length=8)
ax0.legend()
ax0.legend(ncol=2, frameon=False, loc=0)
plt.tight_layout()

save_folder = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0'
#plt.savefig(os.path.join(save_folder, 'SpNR_multiple_neurons_v3.10.pdf'))


#%% Supp Fig 9c and 9d
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from scipy.ndimage import median_filter
def run(num, seed=0, Tinit=20000):
    print(num)
    print(seed)
    np.random.seed(seed)
    frate = 400
    img = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/test_timing_new.npy')
    t_d = img.copy()
    for tp in range(img.shape[0]):
        if tp > 0:
            t_d[tp] = img[tp] - img[tp - 1] + 0.995 * t_d[tp - 1]
    img = -t_d[5000:]
    # img = np.array([img[:20000] for _ in range(5)])
    # img =  img.reshape(-1)
    trace_all = []
    rand = np.random.randint(0, 5000, num) # random translation
    for i in range(num):
        temp = img.copy()
        temp = np.append(temp, img[:rand[i]])
        temp = temp[rand[i]:]        
        trace_all.append(temp)
    trace_all = np.array(trace_all)
    trace_all = trace_all[:60000]
    
    saoz = SignalAnalysisOnlineZ(fr=frate, flip=False, robust_std=False, do_scale=False, detrend=False)
    saoz.fit(trace_all[:,:Tinit], len(img))
    for n in range(Tinit, 60000):
        saoz.fit_next(trace_all[:, n:n+1], n)
    return np.array(saoz.t_detect)[Tinit:] * 1000


tt = run(num=100)
# timing = []
# for N in (500, 200, 100):
#     timing.append(run(num=N))

# runs = 10
# t500 = np.array([run(500, seed=seed) for seed in range(runs)])
# t100 = np.array([run(100, seed=seed) for seed in range(runs)])

#np.save('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/spike_timing_FIOLA_v4.1.npy', [t100, t500, timing])

#%%
t100, t500, timing = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/spike_timing_FIOLA_v4.1.npy', allow_pickle=True)
t100 = t100[:, :]
t500 = t500[:, :]
plt.figure()
for t, l, c in ((t100, '100 neurons', 'orange'), (t500, '500 neurons', 'blue')):
    plt.plot(np.median(t, 0), label=l, color=c, lw=1)
    # plt.fill_between(range(4000), np.percentile(t, 25, 0),
    #                   np.percentile(t, 75, 0), color=c, alpha=.5)
plt.legend(loc=2, frameon=False)
plt.xlabel('Frames')
plt.ylabel('Time (ms)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout(pad=.1)
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_v4.3.pdf')
# plt.figure()
# plt.plot(img[20000:60000])
#saoz_all = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/spike_timing_FIOLA.npy', allow_pickle=True).item()

#%%
import matplotlib.cbook as cbook
data_fr_custom = {}
data_fr_custom[0] = timing[2]
data_fr_custom[1] = timing[1]
data_fr_custom[2] = timing[0]
stats = {}
count = 0
for key in data_fr_custom.keys():
    print(key)
    stats[key] = cbook.boxplot_stats(data_fr_custom[key], labels=str(count))[0]
    stats[key]["q1"], stats[key]["q3"] = np.percentile(
        data_fr_custom[key], [5, 95])
    stats[key]["whislo"], stats[key]["whishi"] = np.percentile(
        data_fr_custom[key], [0.5, 99.5])
    stats[key]["fliers"] = []
    count += 1

colors = ["C0", "C0", "C0"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=range(3),  patch_artist=True)
#ax.set_yscale("log")
ax.set_xticklabels(data_fr_custom.keys())

for median in bplot['medians']:
    median.set_color('black')
    
ax.legend(['FIOLA'])
#ax.set_xticks([])
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Time (ms)')    
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_boxplot_v4.2.pdf')


#########################################################################################################
#########################################################################################################
#%%
#ax.plot(saoz.t_s[:, 10000:].mean(0)/200000)
#plt.title('timing for spike extraction algorithm ')
#plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1/supp/suppl_timing_spike_extraction1.png')
# tt = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/viola_sim5_7_nnls_result.npy')
# trace = tt[:50,:].copy()
# trace = np.repeat(trace, 10, axis=0)
# saoz = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
#                                   detrend=True, flip=True, 
#                                   fr=400,
#                                   adaptive_threshold=True, online_filter_method='median_filter',
#                                   template_window=2, filt_window=15, minimal_thresh=2.8, step=2500, do_plot=False)
# saoz.fit(trace[:, :10000], num_frames=trace.shape[1])
# for n in range(10000, trace.shape[1]):
#     saoz.fit_next(trace[:, n: n+1], n)
#%%
lists = ['454597_Cell_0_40x_patch1_output.npz', '456462_Cell_3_40x_1xtube_10A2_output.npz',
             '456462_Cell_3_40x_1xtube_10A3_output.npz', '456462_Cell_5_40x_1xtube_10A5_output.npz',
             '456462_Cell_5_40x_1xtube_10A6_output.npz', '456462_Cell_5_40x_1xtube_10A7_output.npz', 
             '462149_Cell_1_40x_1xtube_10A1_output.npz', '462149_Cell_1_40x_1xtube_10A2_output.npz', 
             '456462_Cell_4_40x_1xtube_10A4_output.npz', '456462_Cell_6_40x_1xtube_10A10_output.npz',
                 '456462_Cell_5_40x_1xtube_10A8_output.npz', '456462_Cell_5_40x_1xtube_10A9_output.npz', # not make sense
                 '462149_Cell_3_40x_1xtube_10A3_output.npz', '466769_Cell_2_40x_1xtube_10A_6_output.npz',
                 '466769_Cell_2_40x_1xtube_10A_4_output.npz', '466769_Cell_3_40x_1xtube_10A_8_output.npz']
#%%
fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/saoz_test.npy', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/saoz_training.npy', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/volpy_test.npy', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/volpy_training.npy']
mm = []
for ff in fnames:
    m = np.load(ff, allow_pickle=True).item()
    mm.append(m['compound_f1'])    
f1 = np.arange(0, 16)
f2 = np.arange(0, 16) 
test_set = np.array([2, 6, 10, 14])
training_set = np.array([ 0,  1,  3,  4,  5,  7,  8,  9, 11, 12, 13, 15])
mm[1] array([0.85, 0.97, 0.46, 0.9 , 0.65, 0.82, 0.81, 0.68, 0.3 , 0.79, 0.48,
       0.95])
mm[0] array([0.94, 0.75, 0.55, 0.79])
mm[3] array([0.89, 0.96, 0.19, 0.88, 0.66, 0.86, 0.73, 0.75, 0.53, 0.71, 0.85,
       0.9 ])
mm[2] array([0.85, 0.97, 0.7 , 0.88])

volpy = np.array([0.89, 0.96, 0.85, 0.19, 0.88, 0.66, 0.97, 0.86, 0.73, 0.75, 0.7, 0.53, 0.71, 0.85, 0.88, 
       0.9 ])
viola = np.array([0.85, 0.97, 0.94, 0.46, 0.9 , 0.65, 0.75, 0.82, 0.81, 0.68,0.55,  0.3 , 0.79, 0.48, 0.79, 
       0.95])


#%%
#save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
labels = names
#lists1 = [li[:13]   for li in lists]
#labels = lists1

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars

viola = result['viola']['f1']
volpy = result['volpy']['f1']
v = np.array([viola, volpy])
v_mean = v.mean(1)
v_std = v.std(1)


from matplotlib import gridspec
fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
rects1 = ax0.bar(x+1 - width/2, viola, width, label=f'Fiola')
rects2 = ax0.bar(x+1 + width/2, volpy, width, label=f'VolPy')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['bottom'].set_visible(False)
ax0.set_ylim([0,1])
ax0.set_xlabel('Cell')
ax0.set_xticks(x+1)

ax0.set_ylabel('F1 Score')
#ax0.set_title('Fiola vs VolPy')
#ax0.set_xticklabels(labels, rotation='vertical', fontsize=3)
ax0.xaxis.set_ticks_position('none') 
ax0.yaxis.set_tick_params(length=8)
ax0.legend()
ax0.legend(ncol=2, frameon=False, loc=0)
plt.tight_layout()
#plt.savefig(os.path.join(save_folder, 'one_neuron_F1_v2.0.pdf'))

x = np.arange(0, 1)  # the label locations
width = 0.35  # the width of the bars
rects1 = ax1.bar(x - width/2, v_mean[0], width, yerr=v_std[0], capsize=10, label=f'Viola')
rects2 = ax1.bar(x + width/2, v_mean[1], width, yerr=v_std[1], capsize=10, label=f'VolPy')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Average')
ax1.xaxis.set_ticks_position('none') 
ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim([0,1])

#ax1.set_xticks(None)
#ax1.set_xticklabels(labels, rotation='horizontal', fontsize=5)
#ax1.legend()
#save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
#plt.savefig(os.path.join(save_folder, 'one_neuron_F1_average_v2.1.pdf'))


#%%
a = np.zeros((10,10))
a[0,0] = 1
plt.imshow(a)

#%%
from nmf_support import normalize 
fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/overlapping_result.npy',
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/neuron1&2_x[6, -2]_y[11, -3]_0percent_neuron_1_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/neuron1&2_x[6, -2]_y[11, -3]_0percent_neuron_2_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/neuron1&2_x[6, -2]_y[9, -2]_10percent_neuron_1_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/neuron1&2_x[6, -2]_y[9, -2]_10percent_neuron_2_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/neuron1&2_x[5, -2]_y[6, -2]_25percent_neuron_1_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/neuron1&2_x[5, -2]_y[6, -2]_25percent_neuron_2_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/spike_detection_saoz_456462_Cell_3_40x_1xtube_10A2_output.npy', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/spike_detection_saoz_456462_Cell_3_40x_1xtube_10A3_output.npy',
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron_result/456462_Cell_3_40x_1xtube_10A2_output.npz', 
          '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron_result/456462_Cell_3_40x_1xtube_10A3_output.npz']


metrics = np.load(fnames[0], allow_pickle=True)
dict1 = np.load(fnames[9], allow_pickle=True)
dict2 = np.load(fnames[10], allow_pickle=True)
separate1 = np.load(fnames[7], allow_pickle=True).item()
separate2 = np.load(fnames[8], allow_pickle=True).item()
percent_0_1 = np.load(fnames[1], allow_pickle=True).item()
percent_0_2 = np.load(fnames[2], allow_pickle=True).item()
percent_10_1 = np.load(fnames[3], allow_pickle=True).item()
percent_10_2 = np.load(fnames[4], allow_pickle=True).item()
percent_25_1 = np.load(fnames[5], allow_pickle=True).item()
percent_25_2 = np.load(fnames[6], allow_pickle=True).item()

#%%
#%%
folders = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/overlapping_neurons/neuron1&2_x[6, -2]_y[11, -3]_0percent',
           '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/overlapping_neurons/neuron1&2_x[6, -2]_y[9, -2]_10percent',
           '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/original_data/overlapping_neurons/neuron1&2_x[5, -2]_y[6, -2]_25percent',
           '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/456462_Cell_3_40x_1xtube_10A2', 
           '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/456462_Cell_3_40x_1xtube_10A3',
           '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/456462_Cell_3_40x_1xtube_10A2/456462_Cell_3_40x_1xtube_10A2_output.npz',
           '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/456462_Cell_3_40x_1xtube_10A3/456462_Cell_3_40x_1xtube_10A3_output.npz']


def load_viola_result(folder, string='v2.1'):
    viola_folder = os.path.join(folder, 'viola')            
    files = os.listdir(viola_folder)
    files = [os.path.join(viola_folder, file) for file in files if string in file]
    if len(files)>1:
        raise Exception('more than one file exists')
    file = files[0]
    vi = np.load(file, allow_pickle=True).item()
    return vi

percent_0 = load_viola_result(folders[0])
percent_10 = load_viola_result(folders[1])
percent_25 = load_viola_result(folders[2])
separate1 = load_viola_result(folders[3])
separate2 = load_viola_result(folders[4])
dict1 = np.load(folders[5], allow_pickle=True)
dict2 = np.load(folders[6], allow_pickle=True)

#%%
scope = [40500, 41500]
figs, axs = plt.subplots(2,1)
t1 = dict1['v_t'][scope[0]]
t2 = dict1['v_t'][scope[1]]
te1 = np.where(dict1['e_t']>t1)[0][0]
te2 = np.where(dict1['e_t']<t2)[0][-1]


axs[0].plot(dict1['e_t'], normalize(dict1['e_sg'])/2-10, color='black', label='ephys')
axs[0].plot(dict1['v_t'], normalize(separate1.t0[0]), label='neuron1_separate', color='blue')
axs[0].plot(dict1['v_t'], normalize(percent_0.t0[0]), label='neuron1_0_overlapping', color='orange')
axs[0].plot(dict1['v_t'], normalize(percent_10.t0[0]), label='neuron1_10_overlapping', color='red')
axs[0].plot(dict1['v_t'], normalize(percent_25.t0[0]), label='neuron1_25_overlapping', color='green')
axs[0].vlines(dict1['v_t'][separate1.index[0]], 10, 10.5, color='blue')
axs[0].vlines(dict1['v_t'][percent_0.index[0]], 10.5, 11, color='orange')
axs[0].vlines(dict1['v_t'][percent_10.index[0]], 11, 11.5, color='red')
axs[0].vlines(dict1['v_t'][percent_25.index[0]], 11.5, 12, color='green')
axs[0].set_xlim([t1, t2])
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].get_yaxis().set_visible(False)
axs[0].get_xaxis().set_visible(False)


t1 = dict2['v_t'][scope[0]]
t2 = dict2['v_t'][scope[1]]
te1 = np.where(dict2['e_t']>t1)[0][0]
te2 = np.where(dict2['e_t']<t2)[0][-1]
#plt.savefig(os.path.join(save_folder, 'overlapping_neuron1_trace.pdf'))
axs[1].plot(dict2['e_t'], normalize(dict2['e_sg'])/2-10, color='black', label='ephys')
axs[1].plot(dict2['v_t'], normalize(separate2.t0[0]), label='neuron1_separate', color='blue')
axs[1].plot(dict2['v_t'], normalize(percent_0.t0[1]), label='neuron1_0_overlapping', color='orange')
axs[1].plot(dict2['v_t'], normalize(percent_10.t0[1]), label='neuron1_10_overlapping', color='red')
axs[1].plot(dict2['v_t'], normalize(percent_25.t0[1]), label='neuron1_25_overlapping', color='green')
axs[1].vlines(dict2['v_t'][separate2.index[0]], 10, 10.5, color='blue')
axs[1].vlines(dict2['v_t'][percent_0.index[1]], 10.5, 11, color='orange')
axs[1].vlines(dict2['v_t'][percent_10.index[1]], 11, 11.5, color='red')
axs[1].vlines(dict2['v_t'][percent_25.index[1]], 11.5, 12, color='green')
axs[1].set_xlim([t1, t2])
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['left'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].get_yaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)
axs[1].hlines( -15, t1, t1+0.5)
axs[1].text(t1,-20, '0.5s')

save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(save_folder, 'overlapping_neurons_traces_v2.1.pdf'))



#%%


plt.savefig(os.path.join(save_folder, 'overlapping_neuron2_trace.pdf'))

#%%
import caiman as cm
fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron/456462_Cell_3_40x_1xtube_10A3_mc.tif', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/one_neuron/456462_Cell_3_40x_1xtube_10A2_mc.tif', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/overlapping_neurons/neuron1&2_x[6, -2]_y[11, -3]_0percent.tif', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/overlapping_neurons/neuron1&2_x[6, -2]_y[9, -2]_10percent.tif', '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/test_data/overlapping_neurons/neuron1&2_x[5, -2]_y[6, -2]_25percent.tif' ]

for ff in fnames:
    m = cm.load(ff)
    plt.figure()
    plt.imshow(m.mean(axis=0), cmap='gray')
    plt.savefig(os.path.join(save_folder,os.path.split(ff)[-1][:-4]+'_spatial.pdf' ))
    plt.show()

#%%
fnames = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neurons/overlapping_result.npy'

metric = np.load(fnames, allow_pickle=True).item()

0.85, 0.97, 0.94

metric['compound_f1']

F1 = np.array([[0.85, 0.97], [0.7429963459196103, 0.9639614855570839],
 [0.751434611899728, 0.9635136995731792],
 [0.7592814371257486, 0.9598010774968918],
 [0.85, 0.94],
 [0.7602409638554217, 0.9120998372219207],
 [0.7904789891272407, 0.911860718171926],
 [0.8080808080808082, 0.9031550068587106],
 [0.97, 0.94],
 [0.9589268427603375, 0.8631962957766243],
 [0.9567747298420616, 0.8471391972672929],
 [0.9488969561574979, 0.8225221303149035]])
    
F1 = F1.reshape([3,8])



################################################################################################
#%%
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/Figures/overlapping_neurons'
scope = [40500, 41500]

t1 = dict1['v_t'][scope[0]]
t2 = dict1['v_t'][scope[1]]
te1 = np.where(dict1['e_t']>t1)[0][0]
te2 = np.where(dict1['e_t']<t2)[0][-1]
separate1_spike_scope = np.intersect1d(np.where(np.array(separate1['indexes'])>scope[0])[0], np.where(np.array(separate1['indexes'])<scope[1])[-1])
separate1_spikes = dict1['v_t'][separate1['indexes']][separate1_spike_scope]
percent_0_1_spike_scope = np.intersect1d(np.where(np.array(percent_0_1['indexes'])>scope[0])[0], np.where(np.array(percent_0_1['indexes'])<scope[1])[-1])
percent_0_1_spikes = dict1['v_t'][percent_0_1['indexes']][percent_0_1_spike_scope]
percent_10_1_spike_scope = np.intersect1d(np.where(np.array(percent_10_1['indexes'])>scope[0])[0], np.where(np.array(percent_10_1['indexes'])<scope[1])[-1])
percent_10_1_spikes = dict1['v_t'][percent_10_1['indexes']][percent_10_1_spike_scope]
percent_25_1_spike_scope = np.intersect1d(np.where(np.array(percent_25_1['indexes'])>scope[0])[0], np.where(np.array(percent_25_1['indexes'])<scope[1])[-1])
percent_25_1_spikes = dict1['v_t'][percent_25_1['indexes']][percent_25_1_spike_scope]


plt.figure()
plt.plot(dict1['e_t'][te1:te2], normalize(dict1['e_sg'][te1:te2])/2-2, color='black', label='ephys')
plt.plot(dict1['v_t'][scope[0]:scope[1]], normalize(separate1['trace'].flatten())[scope[0]:scope[1]], label='neuron1_separate', color='blue')
plt.plot(dict1['v_t'][scope[0]:scope[1]], normalize(percent_0_1['trace'].flatten())[scope[0]:scope[1]], label='neuron1_0_overlapping', color='orange')
plt.plot(dict1['v_t'][scope[0]:scope[1]], normalize(percent_10_1['trace'].flatten())[scope[0]:scope[1]], label='neuron1_10_overlapping', color='red')
plt.plot(dict1['v_t'][scope[0]:scope[1]], normalize(percent_25_1['trace'].flatten())[scope[0]:scope[1]], label='neuron1_25_overlapping', color='green')
plt.vlines(separate1_spikes, 1, 1.1, color='blue')
plt.vlines(percent_0_1_spikes, 1.1, 1.2, color='orange')
plt.vlines(percent_10_1_spikes, 1.2, 1.3, color='red')
plt.vlines(percent_25_1_spikes, 1.3, 1.4, color='green')
plt.legend()


t1 = dict2['v_t'][scope[0]]
t2 = dict2['v_t'][scope[1]]
te1 = np.where(dict2['e_t']>t1)[0][0]
te2 = np.where(dict2['e_t']<t2)[0][-1]
separate2_spike_scope = np.intersect1d(np.where(np.array(separate2['indexes'])>scope[0])[0], np.where(np.array(separate2['indexes'])<scope[1])[-1])
separate2_spikes = dict2['v_t'][separate2['indexes']][separate2_spike_scope]
percent_0_2_spike_scope = np.intersect1d(np.where(np.array(percent_0_2['indexes'])>scope[0])[0], np.where(np.array(percent_0_2['indexes'])<scope[1])[-1])
percent_0_2_spikes = dict2['v_t'][percent_0_2['indexes']][percent_0_2_spike_scope]
percent_10_2_spike_scope = np.intersect1d(np.where(np.array(percent_10_2['indexes'])>scope[0])[0], np.where(np.array(percent_10_2['indexes'])<scope[1])[-1])
percent_10_2_spikes = dict2['v_t'][percent_10_2['indexes']][percent_10_2_spike_scope]
percent_25_2_spike_scope = np.intersect1d(np.where(np.array(percent_25_2['indexes'])>scope[0])[0], np.where(np.array(percent_25_2['indexes'])<scope[1])[-1])
percent_25_2_spikes = dict2['v_t'][percent_25_2['indexes']][percent_25_2_spike_scope]


plt.figure()
plt.plot(dict2['e_t'][te1:te2], normalize(dict2['e_sg'][te1:te2])/2-2, color='black', label='ephys')
plt.plot(dict2['v_t'][scope[0]:scope[1]], normalize(separate2['trace'].flatten())[scope[0]:scope[1]], label='neuron2_separate', color='blue')
plt.plot(dict2['v_t'][scope[0]:scope[1]], normalize(percent_0_2['trace'].flatten())[scope[0]:scope[1]], label='neuron2_0_overlapping', color='orange')
plt.plot(dict2['v_t'][scope[0]:scope[1]], normalize(percent_10_2['trace'].flatten())[scope[0]:scope[1]], label='neuron2_10_overlapping', color='red')
plt.plot(dict2['v_t'][scope[0]:scope[1]], normalize(percent_25_2['trace'].flatten())[scope[0]:scope[1]], label='neuron2_25_overlapping', color='green')
plt.vlines(separate2_spikes, 1, 1.1, color='blue')
plt.vlines(percent_0_2_spikes, 1.1, 1.2, color='orange')
plt.vlines(percent_10_2_spikes, 1.2, 1.3, color='red')
plt.vlines(percent_25_2_spikes, 1.3, 1.4, color='green')
plt.legend()

#%%
from matplotlib import gridspec
fig = plt.figure(figsize=(8, 6)) 
#gs = gridspec.GridSpec(1, 1) 
#ax0 = plt.subplot(gs[0])
ax1 = plt.subplot()
# rects1 = ax0.bar(x+1 - width/2, viola, width/4, label=f'FIOLA_25ms')
# rects2 = ax0.bar(x+1 - width/4, volpy, width/4, label=f'VolPy')
# rects3 = ax0.bar(x+1  , viola1, width/4, label=f'FIOLA_17.5ms')
# rects4 = ax0.bar(x+1 + width/4, viola2, width/4, label=f'FIOLA_12.5ms')


# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax0.spines['top'].set_visible(False)
# ax0.spines['right'].set_visible(False)
# ax0.spines['bottom'].set_visible(False)
# ax0.set_ylim([0,1])
# ax0.set_xlabel('Cell')
# ax0.set_xticks(x+1)

# ax0.set_ylabel('F1 Score')
# #ax0.set_title('Fiola vs VolPy')
# #ax0.set_xticklabels(labels, rotation='vertical', fontsize=3)
# ax0.xaxis.set_ticks_position('none') 
# ax0.yaxis.set_tick_params(length=8)
# ax0.legend()
# ax0.legend(ncol=2, frameon=False, loc=0)
# plt.tight_layout()
#plt.savefig(os.path.join(save_folder, 'one_neuron_F1_v2.0.pdf'))


x = np.arange(0, 1)  # the label locations
width = 0.35  # the width of the bars
rects1 = ax1.bar(x - width/2, v_mean[0], width/4, yerr=v_std[0], capsize=5, label=f'FIOLA_25ms')
rects2 = ax1.bar(x - width/4, v_mean[1], width/4, yerr=v_std[1], capsize=5, label=f'VolPy')
rects3 = ax1.bar(x  , v_mean[2], width/4, yerr=v_std[2], capsize=5, label=f'FIOLA_17.5ms')
rects4 = ax1.bar(x + width/4, v_mean[3], width/4, yerr=v_std[3], capsize=5, label=f'FIOLA_12.5ms')
rects4 = ax1.bar(x + width/2, v_mean[4], width/4, yerr=v_std[4], capsize=5, label=f'MeansROI')


ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Average')
ax1.xaxis.set_ticks_position('none') 
ax1.yaxis.set_ticks_position('none') 
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylim([0,1])

#ax1.set_xticks(None)
#ax1.set_xticklabels(labels, rotation='horizontal', fontsize=5)
ax1.legend()
#save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
#plt.savefig(os.path.join(save_folder, 'one_neuron_F1_average_v2.1_Fiola&VolPy_non_symm_median_1.pdf'))

#%%
from scipy import stats
rvs1 = stats.norm.rvs(loc=7,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
stats.ttest_ind(volpy,viola2,  equal_var = False)
stats.wilcoxon(volpy, viola2)
stats.ttest_ind(viola,viola2,  equal_var = False)
stats.wilcoxon(viola, viola2)

#%%
for ii, m1 in enumerate([viola, volpy, viola1, viola2]):
    for jj, m2 in enumerate([viola, volpy, viola1, viola2]):
        if ii < jj:
            print(ii); print(jj)
            print(stats.ttest_rel(m1, m2))
            #print(stats.wilcoxon(m1, m2))



#%%
from scipy import stats
rvs1 = stats.norm.rvs(loc=7,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
stats.ttest_ind(viola, meanroi,  equal_var = False)
stats.wilcoxon(viola, meanroi, alternative='greater')
#stats.ttest_ind(viola, viola2,  equal_var = False)
stats.wilcoxon(viola, meanroi)


#spnr_real = np.delete(spnr_real, 3)
#F1_real = np.delete(F1_real, 3)

lr = LinearRegression()
lr.fit(spnr_real[:, np.newaxis], F1_real)

y_real = lr.predict(x_test)

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(min_samples=17, loss='squared_loss')
ransac = RANSACRegressor()
ransac.fit(spnr_real[:, np.newaxis], F1_real)
y_real_ransac = ransac.predict(x_test)
#plt.plot(x_test.flatten(), y_real)

from sklearn.linear_model import HuberRegressor
huber = HuberRegressor(alpha=0.0, epsilon=2.1)
huber.fit(spnr_real[:, np.newaxis], F1_real)
y_real_huber = huber.predict(x_test)

from sklearn.linear_model import TheilSenRegressor
theilsen = TheilSenRegressor()
theilsen.fit(spnr_real[:, np.newaxis], F1_real)
y_real_theilsen = theilsen.predict(x_test)
#%%
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures'
label = ['Cell1/2', 'Cell1/3', 'Cell2/3']
#x = np.arange(len(label))  # the label locations
x = np.array([0, 0.6, 1.2])
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
"""
rects1 = ax.bar(x - width/2, F1[:,1], width/8, label='Separ.', color='blue', edgecolor='black')
rects3 = ax.bar(x - width/4, F1[:,3], width/8, label='0%', color='orange', edgecolor='black')
rects5 = ax.bar(x , F1[:,5], width/8, label='10%', color='red', edgecolor='black')
rects7 = ax.bar(x + width/4, F1[:,7], width/8, label='25%', color='green', edgecolor='black')
rects2 = ax.bar(x - width/2+width/8, F1[:,0], width/8,  color='blue', edgecolor='black')
rects4 = ax.bar(x - width/4+width/8, F1[:,2], width/8,  color='orange', edgecolor='black')
rects6 = ax.bar(x + width/8, F1[:,4], width/8,  color='red', edgecolor='black')
rects8 = ax.bar(x + width/4+width/8, F1[:,6], width/8,  color='green', edgecolor='black')
"""
rects1 = ax.bar(x - width/2, F1[:,0], width/8, label='Separ.', color='blue', edgecolor='black')
rects3 = ax.bar(x - width/4, F1[:,2], width/8, label='0%', color='orange', edgecolor='black')
rects5 = ax.bar(x , F1[:,4], width/8, label='10%', color='red', edgecolor='black')
rects7 = ax.bar(x + width/4, F1[:,6], width/8, label='25%', color='green', edgecolor='black')
rects2 = ax.bar(x - width/2+width/8, F1[:,1], width/8,  color='blue', edgecolor='black')
rects4 = ax.bar(x - width/4+width/8, F1[:,3], width/8,  color='orange', edgecolor='black')
rects6 = ax.bar(x + width/8, F1[:,5], width/8,  color='red', edgecolor='black')
rects8 = ax.bar(x + width/4+width/8, F1[:,7], width/8,  color='green', edgecolor='black')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylabel('F1 Score')
ax.set_title('Scores by neuron group')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.xaxis.set_ticks_position('none') 
ax.yaxis.set_tick_params(length=8)
ax.set_ylim([0,1])
ax.legend(ncol=4, frameon=False, loc='upper center')
plt.tight_layout()
#plt.savefig(os.path.join(save_folder, 'overlapping_neurons_F1_v2.1.pdf' ))

#%%
result_all = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neuron/meanroi_F1_v3.0.npy', allow_pickle=True).item()
F2 = np.zeros([3,8])

for keys, values in result_all.items():
    print(keys)
    if '0&1' in keys:
        row = 0        
    if '0&2' in keys:
        row = 1
    if '1&2' in keys:
        row = 2
    print(row)
    
    if '_25percent' in keys:
        col = 6
    if '_10percent' in keys:
        col = 4
    if '_0percent' in keys:
        col = 2
    print(col)
        
    for kk, vv in values.items():
        F2[row, col] = vv['f1']
        col = col + 1
        
result = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/mean_roi_online_v3.0.npy',
                 allow_pickle=True).item()
rr = result['meanroi_online']['f1'].copy()
F2[0,0] = rr[0]
F2[1,0] = rr[0]
F2[0,1] = rr[1]
F2[1,1] = rr[2]
F2[2,0] = rr[1]
F2[2,1] = rr[2]

ff1 = F2[0].reshape((2,4), order='F')

#%%
#x = np.array([0, 1])
#width = 0.1  # the width of the bars
      
# colors=['blue', 'orange']
# methods = ['FIOLA', 'meanroi online']
# for idx in range(2):
#     ax1.bar(x+width*(idx-0.5), [ff.mean(1), ff1.mean(1)][idx], width, color=colors[idx], label=methods[idx])
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# #ax1.spines['bottom'].set_visible(False)
# ax1.set_ylabel('F1 Score')
# ax1.set_xticks(x)
# ax1.set_xticklabels(label)
# ax1.xaxis.set_ticks_position('none') 
# ax1.yaxis.set_tick_params(length=8)
# ax1.set_ylim([0.5,1])
# ax1.legend(frameon=False)
# fig.tight_layout()

#plt.savefig(save_folder+'v3.0/real_data_overlapping.pdf')

#%%

#%%
from numba import jit
@jit
def speed_up(saoz1, trace, n):
    for n in range(10000, trace.shape[1]):
        saoz1.fit_next(trace[:, n: n+1], n)
        
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
tt = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/viola_sim5_7_nnls_result.npy')
trace = tt[:50,:].copy()
trace = np.repeat(trace, 2, axis=0)
saoz1 = SignalAnalysisOnlineZ(do_scale=False, freq=15, 
                                  detrend=True, flip=True, 
                                  fr=400,
                                  adaptive_threshold=True, online_filter_method='median_filter',
                                  template_window=2, filt_window=15, minimal_thresh=2.8, step=2500, do_plot=False)
saoz1.fit(trace[:, :10000], num_frames=trace.shape[1])
print('processing online')
for n in range(10000, trace.shape[1]):
    saoz1.fit_next(trace[:, n: n+1], n)

#t_detect = np.array(saoz.t_detect[10000:])*1000
t1_detect = np.array(saoz1.t_detect[10000:])*1000
t_detect = t1_detect
    
#%%    
fig, ax = plt.subplots(1,1)
ax.plot(t_detect, label=f'500 neurons', color='orange')
ax.plot(t1_detect, label=f'100 neurons', color='blue')
ax.set_xlabel('# frames (10^4)')
ax.set_ylabel('time (ms)')
ax.legend(frameon=False)
ax.set_xticks(np.arange(0, 70000, 10000))
ax.set_xticklabels(['0', '1', '2', '3', '4', '5', '6'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.hlines(0.7, 0, 65000, color='black', linestyle='--')
ax.xaxis.set_tick_params(length=8)
ax.yaxis.set_tick_params(length=8)
ax.set_ylim(0, 1)



t1 = np.array(saoz_all[100][20000:]) * 1000
t2 = np.array(saoz_all[200][20000:]) * 1000
t3 = np.array(saoz_all[500][20000:]) * 1000

plt.figure()
plt.bar([0, 1, 2], [t1.mean(), t2.mean(), t3.mean()], yerr=[t1.std(), t2.std(), t3.std()])
plt.ylabel('Timing (ms)')
#plt.xlabel('Number of neurons')
plt.xticks([0, 1, 2], ['100 neurons', '200 neurons', '500 neurons'])
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_mean_v4.0pdf')
#%%
plt.figure()
#plt.plot(saoz.t_detect)
plt.plot(np.array(saoz_all[500])[20000:]*1000, label='500 neurons', color='orange')
plt.plot(np.array(saoz_all[100])[20000:]*1000, label='100 neurons', color='blue')
plt.ylabel('Timing (ms)')
plt.legend()
#plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_v4.0.pdf')
