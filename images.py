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
lists1 = [li[:13]   for li in lists]
labels = lists1
volpy = np.array([0.95, 0.94, 0.7,  0.35, 0.91, 0.72, 0.97, 0.79])
online = np.array([0.98, 0.99, 0.88, 0.87, 0.88, 0.32, 0.89, 0.76])
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, volpy, width, label=f'VolPy:avgF1={volpy.mean().round(2)}')
rects2 = ax.bar(x + width/2, online, width, label=f'Online:avgF1={online.mean().round(2)}')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Scores')
ax.set_title('Online vs VolPy')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation='horizontal', fontsize=5)
ax.legend()
plt.savefig('/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/picture/sao_accuracy.pdf')



