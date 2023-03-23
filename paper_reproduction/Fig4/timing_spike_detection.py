#!/usr/bin/env python
#%%
import sys
sys.path.append('/home/nel/CODE/VIOLA')
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size': 7, 
                     'ytick.major.size': 7})
import numpy as np
import pyximport
pyximport.install()
from scipy.ndimage import gaussian_filter
from random import sample

from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter

def voltage_trace_sim(num, T, nSpikes, frate):
    ISI = 0.01
    spikeSize = 3
    spikeAmp = 0.15
    sn = 0.01
    subThresh = np.zeros((T, num))
    trace = np.zeros((T, num))
    spikes = []    
    simSpike = np.array([0,-0.00610252301264466,0.0153324173640442,0.0394596567723412,0.0550890774222366,0.0855483409375236,0.109121613111175,0.159347286748449,0.272280693885892,0.454264679126972,1,0.678615836282434,0.441859762420943,0.232255883375386,0.0936575998651618,-0.0134953901062200,-0.0507224917483669,-0.0559150686307210,-0.0532159906094771,-0.0212346446514174,0])
    
    # subthreshold
    for n in range(num):
        subThresh[:,n] = gaussian_filter(np.maximum(-2,np.random.randn(1, T)), 0.02*frate)
        subThresh[:,n] = subThresh[:,n]/np.percentile(subThresh[:,n],95)  
        subThresh[:,n] = subThresh[:,n]-np.percentile(subThresh[:,n],5)
    
    # spikes
    for n in range(num):
        # spike time
        vTimes = np.cumsum(np.random.randint(int(ISI*frate), int((2*ISI*frate)-1), size=(1, T))) 
        vTimes = vTimes[vTimes>len(simSpike)]
        vTimes = vTimes[vTimes<(T-len(simSpike))]                  
        ST = np.unique(np.random.choice(vTimes, nSpikes))
        spikes.append(ST)
        
        # convolve with spike
        t2 = np.zeros((T))
        t2[ST] = 1
        t2 = np.convolve(t2, simSpike, 'same')
        
        # construct the trace
        tmpTrace=(subThresh[:,n]/spikeSize + t2);
        tmpTrace = tmpTrace - np.median(tmpTrace)
        trace[:,n] = 1 + (tmpTrace*spikeAmp)   
    
    # noise
    trace += sn * np.random.randn(T, num).astype(np.float32)
    return trace, spikes


#%%
#path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_debug_spike_detection/'
path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/'
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
def run(num, seed=0, Tinit=20000):
    print(f'start running {num}')
    np.random.seed(seed)
    frate = 400
    T = 50000
    nSpikes = 125
    
    trace, _ = voltage_trace_sim(num, T=T, nSpikes=nSpikes, frate=frate)
    trace = trace.T
    saoz = SignalAnalysisOnlineZ(fr=frate, flip=False, robust_std=False, do_scale=False, detrend=False)
    saoz.fit(trace[:,:Tinit], T)
    for n in range(Tinit, T):
        saoz.fit_next(trace[:, n:n+1], n)
    return np.array(saoz.t_detect)[Tinit:] * 1000

runs = 10
t100 = np.array([run(100, seed=seed) for seed in range(runs)])
t500 = np.array([run(500, seed=seed) for seed in range(runs)])

timing = []
for N in (100, 200, 500):
    timing.append(run(num=N))

#np.save(path + 'timing_spike_detection_new1.npy', [t100, t500, timing])

#%%

#save_path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/'
#t100, t500, timing = np.load(path + 'timing_spike_detection.npy', allow_pickle=True)
#t100, t500, timing = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/timing_spike_detection_new.npy', allow_pickle=True)
#t100 = t100[:, :]
#t500 = t500[:, :]
plt.figure()
for t, l, c in ((t500, '500 neurons', 'orange'), (t100, '100 neurons', 'blue')):
    plt.plot(np.median(t, 0), label=l, color=c, lw=1)
    # plt.fill_between(range(4000), np.percentile(t, 25, 0),
    #                   np.percentile(t, 75, 0), color=c, alpha=.5)
plt.legend(loc=2, frameon=False)
plt.ylabel('Time (ms)')
plt.plot(range(4000), [0.05] * 4000, color='black')
plt.text(2000, 0.05, '10s', color='black')
plt.gca().spines['bottom'].set_visible(False)
plt.gca().set_xticks([])
plt.tight_layout(pad=.1)
#plt.savefig(save_path+'Fig_supp_detection_timing_v4.5.pdf')
#%%
# # %%
import matplotlib.cbook as cbook
data_fr_custom = {}
data_fr_custom[0] = timing[0]
data_fr_custom[1] = timing[1]
data_fr_custom[2] = timing[2]
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
ax.set_xticks([])
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Time (ms)')    
plt.savefig(save_path+'Fig_supp_detection_timing_boxplot_v4.5.pdf')


#%%
frate=400
saoz_all = {}
for num_neurons in [100, 200, 500]:
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

path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction'
#np.save(path+'/timing_voltage', saoz_all)

#%%
plt.figure()
#plt.plot(saoz.t_detect)
plt.plot(np.array(saoz_all[500].t_detect)[20000:]*1000, label='500 neurons', color='orange')
plt.plot(np.array(saoz_all[100].t_detect)[20000:]*1000, label='100 neurons', color='blue')
plt.ylabel('Timing (ms)')
plt.legend()
plt.savefig(path+'/Fig_supp_detection_timing_v3.7.pdf')

#%%
t1 = np.array(saoz_all[100].t_detect[20000:]) * 1000
t2 = np.array(saoz_all[200].t_detect[20000:]) * 1000
t3 = np.array(saoz_all[500].t_detect[20000:]) * 1000

plt.figure()
plt.bar([0, 1, 2], [t1.mean(), t2.mean(), t3.mean()], yerr=[t1.std(), t2.std(), t3.std()])
plt.ylabel('Timing (ms)')
plt.xlabel('Number of neurons')
plt.xticks([0, 1, 2], ['100 neurons', '200 neurons', '500 neurons'])
plt.savefig(path+'/Fig_supp_detection_timing_mean_v3.7.pdf.pdf')


#%%

from nmf_support import normalize
plt.plot(dict1['v_t'], saoz.t_detect)
plt.plot(dict1['e_t'], dict1['e_sg']/100000)


    
    
    
#%% Supp Fig 9c and 9d
# path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_debug_spike_detection/'

# from fiola.signal_analysis_online import SignalAnalysisOnlineZ
# def run(num, seed=0, Tinit=20000):
#     print(num)
#     frate = 400
#     img = np.load('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/voltage_data/one_neuron/test_timing.npy')
#     img = np.hstack([img[:20000] for _ in range(5)])
#     trace_all = np.stack([img for i in range(num)])
        
#     saoz = SignalAnalysisOnlineZ(fr=frate, flip=False, robust_std=False, do_scale=False, detrend=False)
#     saoz.fit(trace_all[:,:Tinit], len(img))
#     for n in range(Tinit, 100000):
#         saoz.fit_next(trace_all[:, n:n+1], n)
#     return np.array(saoz.t_detect)[Tinit:] * 1000

# runs = 10
# t100 = np.array([run(100, seed=seed) for seed in range(runs)])
# t500 = np.array([run(500, seed=seed) for seed in range(runs)])

# timing = []
# for N in (100, 200, 500):
#     timing.append(run(num=N))

# np.save(path + 'timing_new.npy', [t100, t500, timing])
"""

