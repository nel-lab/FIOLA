import sys
sys.path.append('/home/nel/CODE/VIOLA')
import matplotlib.pyplot as plt
import numpy as np
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
from caiman.source_extraction.cnmf.oasis import OASIS
from time import time

import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False, 
                     'xtick.major.size': 7, 
                     'ytick.major.size': 7})


# %% Run and time
def run(N=100, p=1, new=True, seed=0, T=10000, Tinit=5000, sn=.1, firerate=.2, frate=30):
    # generate data
    np.random.seed(seed)
    traces = (np.random.rand(N, T) < firerate / frate).astype(np.float32)
    if p == 1:
        g = .95
        for i in range(1, T):
            traces[:, i] += g * traces[:, i-1]
    elif p == 2:
        g1, g2 = 1.45, -.475  # d,r = .95,.5
        for i in range(2, T):
            traces[:, i] += g1 * traces[:, i-1] + g2 * traces[:, i-2]
    traces += sn * np.random.randn(N, T).astype(np.float32)
    # deconvolve
    if new: # FIOLA
        saoz = SignalAnalysisOnlineZ(
            mode='calcium', p=p, flip=False, detrend=False)
        saoz.fit(traces[:, :Tinit], T) # initial batch
        for n in range(Tinit, T): # online frame by frame
            saoz.fit_next(traces[:, n:n+1], n)
        return np.array(saoz.t_detect)[Tinit:] * 1000
    else: # CaImAn
        results_foopsi = map(lambda t: constrained_foopsi(t, p=p), traces[:, :Tinit])
        OASISinstances = [OASIS(g=gam[0], lam=lam, b=bl, g2=0 if len(gam) < 2 else gam[1])
                                for _, bl, _, gam, sn, _, lam in results_foopsi]
        for o, t in zip(OASISinstances, traces[:, :Tinit]):
            o.fit(t) # initial batch
        t_detect = []
        for trace_in in traces[:, Tinit:].T:
            t_start = time()
            for o, t in zip(OASISinstances, trace_in):
                o.fit_next(t)  # online frame by frame    
            t_detect.append(time() - t_start)
        return np.array(t_detect) * 1000

timing = []
for N in (100, 200, 500):
    print(f'now processing {N}')
    for p in (1, 2):
        for new in (True, False):
            timing.append(run(N, p, new, T=10000))
            
runs = 10
t500 = np.array([run(500, seed=seed, T=10000) for seed in range(runs)])
t100 = np.array([run(100, seed=seed, T=10000) for seed in range(runs)])



path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction'
np.savez_compressed(path + '/timing_deconvolution_new.npz', timing=timing, t100=t100,t500=t500)
#np.save('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction/calcium_new.npy', 
#        [timing, t500, t100])

#%% Plot
# Supp Fig 9a and 9b
path = '/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/result/test_speed_spike_extraction'
result = np.load(path + '/timing_deconvolution_new.npz', allow_pickle=True)
t100 = result['t100']
t500 = result['t500']
timing = result['timing']

#%% Plot
plt.figure()
for t, l, c in ((t100, '100 neurons', 'blue'), (t500, '500 neurons', 'orange')):
    plt.plot(np.median(t, 0), label=l, color=c, lw=1)
plt.plot(range(900), [0.02] * 900, color='black')
plt.text(450, 0.02, '30s', color='black')
plt.gca().spines['bottom'].set_visible(False)
plt.gca().set_xticks([])
    #plt.fill_between(range(5000), np.percentile(t, 25, 0),
    #                  np.percentile(t, 75, 0), color=c, alpha=.5)
plt.legend(loc=2, frameon=False)
#plt.xlabel('Frames')
plt.ylabel('Time (ms)')
plt.tight_layout(pad=.1)
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_timing_deconv_quartiles_v4.2.pdf')

#%%
import matplotlib.cbook as cbook
data_fr_custom = {}
for i in range(12):
    data_fr_custom[i] = timing[i]
stats = {}
count = 0
for key in data_fr_custom.keys():
    print(key)
    stats[key] = cbook.boxplot_stats(data_fr_custom[key])[0]#, labels=str(count))[0]
    stats[key]["q1"], stats[key]["q3"] = np.percentile(
        data_fr_custom[key], [5, 95])
    stats[key]["whislo"], stats[key]["whishi"] = np.percentile(
        data_fr_custom[key], [0.5, 99.5])

    stats[key]["fliers"] = []
    count += 1

colors = ["C0", "C1", "C0", "C1", "C0", "C1", "C0", "C1", "C0", "C1", "C0", "C1"]
fig, ax = plt.subplots(1, 1)
bplot = ax.bxp(stats.values(),  positions=sum([[i, i+0.5] for i in range(0, 12, 2)], []),  patch_artist=True)
ax.set_xticklabels(data_fr_custom.keys())

for patch, color in zip(bplot["boxes"], colors):
    print(color)
    patch.set_facecolor(color)
    
for median in bplot['medians']:
    median.set_color('black')
    
ax.legend(['FIOLA', 'CaImAn'])
ax.set_xticks([])
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Time (ms)')    
plt.savefig('/media/nel/storage/NEL-LAB Dropbox/NEL/Papers/VolPy_online/figures/v3.0/supp/Fig_supp_detection_timing_calcium_boxplot_v4.2.pdf')

####################################################################################################

"""
#%%
if False:
    plt.figure()
    for t, l, c in ((t100, '100 neurons', 'orange'), (t500, '500 neurons', 'blue')):
        plt.plot(t.mean(0), label=l, color=c, lw=1)
        plt.fill_between(range(5000), (t.mean(0)-t.std(0)/np.sqrt(runs-1)),
                         (t.mean(0)+t.std(0)/np.sqrt(runs-1)), color=c, alpha=.3)
    plt.legend(loc=2, frameon=False)
    plt.xlabel('Frames')
    plt.ylabel('Time (ms)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout(pad=.1)
    #plt.savefig('timing_deconv_mean+-SEM.pdf')
    
    plt.figure()
    for t, l, c in ((t100, '100 neurons', 'orange'), (t500, '500 neurons', 'blue')):
        plt.plot(np.median(t, 0), label=l, color=c, lw=1)
        # plt.fill_between(range(5000), np.percentile(t, 25, 0),
        #                  np.percentile(t, 75, 0), color=c, alpha=.5)
    plt.legend(loc=2, frameon=False)
    #plt.xlim([0, 50000])
    plt.ylim([0, 0.2])
    plt.xlabel('Frames')
    plt.ylabel('Time (ms)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout(pad=.1)
    #plt.savefig('timing_deconv_quartiles.pdf')
    
    
    plt.figure(figsize=(6.4, 4.8))
    plt.bar(np.arange(-.2, 5), np.mean(timing, 1)[::2], yerr=np.std(timing, 1)[::2],
            color='C0', label='FIOLA', width=.4)
    plt.bar(np.arange(.2, 6), np.mean(timing, 1)[1::2], yerr=np.std(timing, 1)[1::2],
            color='C1', label='CaImAn', width=.4)
    plt.legend(loc=2, frameon=False)
    plt.xticks(range(6), ['AR1', 'AR2', 'AR1', 'AR2', 'AR1', 'AR2'])
    plt.xlabel('100 neurons' + ' '*5 + '200 neurons' + ' '*5 + '500 neurons')
    plt.ylabel('Time (ms)')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout(pad=.1)
    #plt.savefig('timing_deconv.pdf')
    
    
"""
    
    
