#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:35:12 2020
accuracy and timing for the algorithm sao
@author: @caichangjia
"""

#%%
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

#%% timing for the algorithm
img = img.astype(np.float32)
sao = SignalAnalysisOnline(thresh_STD=None)
#trace = img[np.newaxis, :]
trace = np.array([img for i in range(50)])
sao.fit(trace[:, :20000], num_frames=100000)
for n in range(20000, img.shape[0]):
    sao.fit_next(trace[:, n: n+1], n)
indexes = np.array((list(set(sao.index[0]) - set([0]))))  

#%%    
plt.figure()
plt.plot(sao.t_detect, label=f'avg:{np.mean(np.array(sao.t_detect)).round(4)}s')
plt.legend()
plt.xlabel('# frames')
plt.ylabel('seconds(s)')
plt.title('timing for spike extraction algorithm 50 neurons')
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/sao_50neurons.pdf')

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
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
#plt.savefig(os.path.join(save_folder, 'one_neuron_F1_average_v2.1.pdf'))

#%%
#save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
labels = names
#lists1 = [li[:13]   for li in lists]
#labels = lists1

x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars

result = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500.npy', allow_pickle=True).item()
viola = result['viola']['f1']
volpy = result['volpy']['f1']
viola1 = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500_filt_window_9_template_window_0.npy', allow_pickle=True)
viola1 = viola1.item()['viola']['f1']

v = np.array([viola, volpy, viola1])
v_mean = v.mean(1)
v_std = v.std(1)

from matplotlib import gridspec
fig = plt.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(1, 2, width_ratios=[9, 1]) 
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
rects1 = ax0.bar(x+1 - width/2, viola, width/2, label=f'Fiola_25ms')
rects2 = ax0.bar(x+1 , volpy, width/2, label=f'VolPy')
rects3 = ax0.bar(x+1 + width/2, viola1, width/2, label=f'Fiola_12.5ms')

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
rects1 = ax1.bar(x - width/2, v_mean[0], width/2, yerr=v_std[0], capsize=5, label=f'Fiola_25ms')
rects2 = ax1.bar(x , v_mean[1], width/2, yerr=v_std[1], capsize=5, label=f'VolPy')
rects3 = ax1.bar(x + width/2, v_mean[2], width/2, yerr=v_std[2], capsize=5, label=f'Fiola_12.5ms')


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
save_folder = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v2.1'
plt.savefig(os.path.join(save_folder, 'one_neuron_F1_average_v2.1_Fiola_25ms_12.5ms.pdf'))

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


#%%
result_all = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_overlapping_neuron/viola_F1_v2.1.npy', 
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
        
result = np.load('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_one_neuron/viola_volpy_F1_v2.1_freq_15_thresh_factor_step_2500.npy',
                 allow_pickle=True).item()
rr = result['viola']['f1'].copy()
F1[0,0] = rr[0]
F1[1,0] = rr[0]
F1[0,1] = rr[1]
F1[1,1] = rr[2]
F1[2,0] = rr[1]
F1[2,1] = rr[2]

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
plt.savefig(os.path.join(save_folder, 'overlapping_neurons_F1_v2.1.pdf' ))

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


