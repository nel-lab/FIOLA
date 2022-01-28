import matplotlib.pyplot as plt
import numpy as np
from fiola.signal_analysis_online import SignalAnalysisOnlineZ

plt.rc('font', size=16)
plt.rc('legend', **{'fontsize': 16})


# %% Run and time
def run(N=100, p=1, use_numba=True, seed=0, T=10000, sn=.1, firerate=.2, frate=30):
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
    saoz = SignalAnalysisOnlineZ(
        mode='calcium', p=p, use_numba=use_numba, nb=0)
    saoz.fit(traces[:, :5000], T)
    for n in range(5000, T):
        saoz.fit_next(traces[:, n:n+1], n)
    return np.array(saoz.t_detect)[5000:] * 1000


runs = 10
t500 = np.array([run(500, seed=seed) for seed in range(runs)])
t100 = np.array([run(100, seed=seed) for seed in range(runs)])

timing = []
for N in (100, 200, 500):
    for p in (1, 2):
        for use_numba in (True, False):
            timing.append(run(N, p, use_numba))

np.savez_compressed('timing.npz', timing=timing, t100=t100,t500=t500)


#%% Plot
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
plt.savefig('timing_deconv_mean+-SEM.pdf')

plt.figure()
for t, l, c in ((t100, '100 neurons', 'orange'), (t500, '500 neurons', 'blue')):
    plt.plot(np.median(t, 0), label=l, color=c, lw=1)
    plt.fill_between(range(5000), np.percentile(t, 25, 0),
                     np.percentile(t, 75, 0), color=c, alpha=.5)
plt.legend(loc=2, frameon=False)
plt.xlabel('Frames')
plt.ylabel('Time (ms)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout(pad=.1)
plt.savefig('timing_deconv_quartiles.pdf')


plt.figure(figsize=(6.4, 4.8))
plt.bar(np.arange(-.2, 5), np.mean(timing, 1)[::2], yerr=np.std(timing, 1)[::2],
        color='C0', label='FIOLA (Numba)', width=.4)
plt.bar(np.arange(.2, 6), np.mean(timing, 1)[1::2], yerr=np.std(timing, 1)[1::2],
        color='C1', label='CaImAn (Cython)', width=.4)
plt.legend(loc=2, frameon=False)
plt.xticks(range(6), ['AR1', 'AR2', 'AR1', 'AR2', 'AR1', 'AR2'])
plt.xlabel('100 neurons' + ' '*5 + '200 neurons' + ' '*5 + '500 neurons')
plt.ylabel('Time (ms)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout(pad=.1)
plt.savefig('timing_deconv.pdf')
