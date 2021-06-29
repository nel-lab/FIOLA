#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:46:15 2020

@author: agiovann
"""
#%%
import time
import numpy as np
from scipy import stats
import time 
from multiprocessing import Queue
from threading import Thread
from multiprocessing import Process as Thread
from multiprocessing import Pool, Semaphore
import pylab as plt
from scipy.signal import argrelextrema
import os 
#from detect_spikes_exceptionality import estimate_running_std
#%%
base_folder = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new',
               '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/data_new'][0]
lists = ['454597_Cell_0_40x_patch1_output.npz', '456462_Cell_3_40x_1xtube_10A2_output.npz',
             '456462_Cell_3_40x_1xtube_10A3_output.npz', '456462_Cell_5_40x_1xtube_10A5_output.npz',
             '456462_Cell_5_40x_1xtube_10A6_output.npz', '456462_Cell_5_40x_1xtube_10A7_output.npz', 
             '462149_Cell_1_40x_1xtube_10A1_output.npz', '462149_Cell_1_40x_1xtube_10A2_output.npz', ]
file_list = [os.path.join(base_folder, file)for file in lists]
dict1 = np.load(file_list[0], allow_pickle=True)
img = dict1['v_sg'][:50000]
#img = img.astype(np.float32)
#img /= estimate_running_std(img, q_min=0.1, q_max=99.9)

#%%
def compute_thresh(peak_height, prev_thresh=None, delta_max=0.03, number_maxima_before=1):
    """ compute threshold by identifying the minima in hostogram between two supposed clusters
    Args:
        peak_height: ndarray
            set of peaks
        
        prev_thresh: float
            threshold at previous iteration
        
        delta_max: float
            max allowed change in threshold
            
        number_maxima_before: int
            number of maxima in second derivative of PDF to take before local 
            minima of PDF. if 0 local minima of PDF is used
        
    Returns:
        thresh: float
            new threshold
        
    """
    kernel = stats.gaussian_kde(peak_height)
    x_val = np.linspace(0,np.max(peak_height),1000)
    pdf = kernel(x_val)
    second_der = np.diff(pdf,2)
    mean = np.mean(peak_height)
    min_idx = argrelextrema(kernel(x_val), np.less)
    minima = x_val[min_idx]
    minima = minima[minima>mean]
    minima_2nd = argrelextrema(second_der, np.greater)
    minima_2nd = x_val[minima_2nd]

    if prev_thresh is None:
        delta_max = np.inf
        prev_thresh = mean                        
             
    if (len(minima)>0) and (np.abs(minima[0]-prev_thresh) < delta_max):
        thresh = minima[0]
        if number_maxima_before>0:
            mnt = (minima_2nd-thresh)
            mnt = mnt[mnt<0]
            thresh += mnt[np.maximum(-len(mnt)+1,-number_maxima_before)]   
    else:
        thresh = prev_thresh
        
    return thresh, pdf, x_val
            
def thread_compute_thresh(queue_in, queue_out, num_timesteps, delta_max, 
                          number_maxima_before, sema):
   os.nice(19)
   peak_height = []
   prev_thresh = None
   while True:       
       el = queue_in.get()
       if el is None:
           break
       peak_height += list(el)
       if len(peak_height)>=num_timesteps:           
           sema.acquire()
           (prev_thresh, pdf, x_val) = compute_thresh(peak_height, prev_thresh=prev_thresh, 
                                        delta_max=delta_max, 
                                        number_maxima_before=number_maxima_before)
           queue_out.put((prev_thresh, pdf, x_val))
           peak_height = []
           sema.release()
#           print('ADDED:'+ str(time.time()-t0))
   
   print('Finished')  
   return None
#%%
if __name__ == "__main__":
    num_timesteps = 20000
    delta_max = 0.03
    number_maxima_before = 1

    if True:
        times_tot = []
        threads = []        
        queues_in = []
        queues_out = [] 
        num_proc = 3
        num_neurons = 50
        sema = Semaphore(num_proc)
        
        for i in range(num_neurons):            
            queues_in.append(Queue(maxsize=-1))
            queues_out.append(Queue(maxsize=-1))
        
#        p = Pool(num_proc)
#        p.starmap_async(thread_compute_thresh, [(queues_in[i], queues_out[i], 
#                                                num_timesteps, delta_max, 
#                                                number_maxima_before) for i in range(num_neurons)])
        
            t = Thread(target=thread_compute_thresh, args=(queues_in[i], queues_out[i], 
                                                           num_timesteps, delta_max, 
                                                           number_maxima_before, sema))
            threads.append(t)
            t.start()

# 
        t0 = time.time()
        pdfs = [[] for i in range(num_neurons)]
        count_update = 0
        for i in range(len(img)):
            np.sum(np.random.random(45000))
            if i%5000 ==0: 
                print(i)                                 
                for j in range(num_neurons):
                    if not queues_out[j].empty():
                        pdfs[j].append(queues_out[j].get())
                        count_update += 1
                        print('updated!'+str(count_update))
                        
            if i%2500==0:                           
                for j in range(num_neurons):                        
                    if queues_in[j].full():
                        print('FULL!')
                    else:
                        queues_in[j].put(img[i-2500:i])

            times_tot.append(time.time()-t0)
        
        for j in range(num_neurons):
            queues_in[j].put(None)
            
        print('**' + str(time.time()-t0))

plt.plot(np.diff(times_tot),'.')
plt.pause(5)