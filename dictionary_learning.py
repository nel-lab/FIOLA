#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:34:51 2020
This file uses spams package for nmf and nnsc
@author: Andrea Giovannucci
"""
import caiman as cm
import pylab as plt
import numpy as np
from caiman.summary_images import local_correlations_movie_offline
import spams
from sklearn.decomposition import NMF, PCA
from caiman.base.movies import to_3D
from scipy import zeros, signal, random
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
import scipy.ndimage as nd
import scipy.sparse as spr

#%%
#def hals(Y, A, C, b, f, bSiz=3, maxIter=5):
#    """ Hierarchical alternating least square method for solving NMF problem
#
#    Y = A*C + b*f
#
#    Args:
#       Y:      d1 X d2 [X d3] X T, raw data.
#           It will be reshaped to (d1*d2[*d3]) X T in this
#           function
#
#       A:      (d1*d2[*d3]) X K, initial value of spatial components
#
#       C:      K X T, initial value of temporal components
#
#       b:      (d1*d2[*d3]) X nb, initial value of background spatial component
#
#       f:      nb X T, initial value of background temporal component
#
#       bSiz:   int or tuple of int
#        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
#        be convolved with each neuron's initial spatial component, then all nonzero
#       pixels will be picked as pixels to be updated, and the rest will be
#       forced to be 0.
#
#       maxIter: maximum iteration of iterating HALS.
#
#    Returns:
#        the updated A, C, b, f
#
#    Authors:
#        Johannes Friedrich, Andrea Giovannucci
#
#    See Also:
#        http://proceedings.mlr.press/v39/kimura14.pdf
#    """
#
#    # smooth the components
#    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
#    K = A.shape[1]  # number of neurons
#    nb = b.shape[1]  # number of background components
##    if bSiz is not None:
##        if isinstance(bSiz, (int, float)):
##             bSiz = [bSiz] * len(dims)
##        ind_A = nd.filters.uniform_filter(np.reshape(A,
##                dims + (K,), order='F'), size=bSiz + [0])
##        ind_A = np.reshape(ind_A > 1e-10, (np.prod(dims), K), order='F')
##    else:
#    ind_A = A>0
#
#    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels
#
#    def HALS4activity(Yr, A, C, iters=2):
#        U = A.T.dot(Yr)
#        V = A.T.dot(A) + np.finfo(A.dtype).eps
#        for _ in range(iters):
#            for m in range(len(U)):  # neurons and background
#                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
#                               V[m, m], 0, np.inf)
#        return C
#
#    def HALS4shape(Yr, A, C, iters=2):
#        U = C.dot(Yr.T)
#        V = C.dot(C.T) + np.finfo(C.dtype).eps
#        for _ in range(iters):
#            for m in range(K):  # neurons
#                ind_pixels = np.squeeze(ind_A[:, m].toarray())
#                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
#                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
#                                            V[m, m]), 0, np.inf)
#            for m in range(nb):  # background
#                A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] - V[K + m].dot(A.T)) /
#                                                     V[K + m, K + m]), 0, np.inf)
#        return A
#
#    Ab = np.c_[A, b]
#    Cf = np.r_[C, f.reshape(nb, -1)]
#    for _ in range(maxIter):
##        Ab = nd.filters.median_filter(np.reshape(Ab,
##                dims + (K+nb,), order='F'), size=bSiz)
#        print(Ab.shape)
#        Ab = nd.filters.gaussian_filter(np.reshape(Ab,
#                dims + (K+nb,), order='F'), sigma=bSiz)
#        import pdb
#        pdb.set_trace()
#        plt.imshow(np.reshape(Ab, dims + (K+nb,), order='F').sum(axis=-1))
#        plt.pause(1)
##        ind_A = np.reshape(ind_A > 0, (np.prod(dims), K), order='F')
##        ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels
##        Cf = HALS4activity(np.reshape(
##            Y, (np.prod(dims), T), order='F'), Ab, Cf)
#        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)
#        
#
#    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)
#%%
def hals(Y, A, C, b, f, bSiz=3, filsiz=(3,3), maxIter=5):
    """ Hierarchical alternating least square method for solving NMF problem

    Y = A*C + b*f

    Args:
       Y:      d1 X d2 [X d3] X T, raw data.
           It will be reshaped to (d1*d2[*d3]) X T in this
           function

       A:      (d1*d2[*d3]) X K, initial value of spatial components

       C:      K X T, initial value of temporal components

       b:      (d1*d2[*d3]) X nb, initial value of background spatial component

       f:      nb X T, initial value of background temporal component

       bSiz:   int or tuple of int
        blur size. A box kernel (bSiz X bSiz [X bSiz]) (if int) or bSiz (if tuple) will
        be convolved with each neuron's initial spatial component, then all nonzero
       pixels will be picked as pixels to be updated, and the rest will be
       forced to be 0.

       maxIter: maximum iteration of iterating HALS.

    Returns:
        the updated A, C, b, f

    Authors:
        Johannes Friedrich, Andrea Giovannucci

    See Also:
        http://proceedings.mlr.press/v39/kimura14.pdf
    """

    # smooth the components
    dims, T = np.shape(Y)[:-1], np.shape(Y)[-1]
    K = A.shape[1]  # number of neurons
    nb = b.shape[1]  # number of background components
    if bSiz is not None:
        if isinstance(bSiz, (int, float)):
             bSiz = [bSiz] * len(dims)
        ind_A = nd.filters.uniform_filter(np.reshape(A,
                dims + (K,), order='F'), size=bSiz + [0])
        ind_A = np.reshape(ind_A > 0, (np.prod(dims), K), order='F')
    else:
        ind_A = A>0

    ind_A = spr.csc_matrix(ind_A)  # indicator of nonnero pixels

    def HALS4activity(Yr, A, C, iters=2):
        U = A.T.dot(Yr)
        V = A.T.dot(A) + np.finfo(A.dtype).eps
        for _ in range(iters):
            for m in range(len(U)):  # neurons and background
                C[m] = np.clip(C[m] + (U[m] - V[m].dot(C)) /
                               V[m, m], 0, np.inf)
        return C

    def HALS4shape(Yr, A, C, iters=2):
        U = C.dot(Yr.T)
        V = C.dot(C.T) + np.finfo(C.dtype).eps
        for _ in range(iters):
            for m in range(K):  # neurons
                ind_pixels = np.squeeze(ind_A[:, m].toarray())
                A[ind_pixels, m] = np.clip(A[ind_pixels, m] +
                                           ((U[m, ind_pixels] - V[m].dot(A[ind_pixels].T)) /
                                            V[m, m]), 0, np.inf)
            for m in range(nb):  # background
                A[:, K + m] = np.clip(A[:, K + m] + ((U[K + m] - V[K + m].dot(A.T)) /
                                                     V[K + m, K + m]), 0, np.inf)
        return A

    Ab = np.c_[A, b]
    Cf = np.r_[C, f.reshape(nb, -1)]
    for _ in range(maxIter):
        Cf = HALS4activity(np.reshape(
            Y, (np.prod(dims), T), order='F'), Ab, Cf)
        Ab = HALS4shape(np.reshape(Y, (np.prod(dims), T), order='F'), Ab, Cf)

    return Ab[:, :-nb], Cf[:-nb], Ab[:, -nb:], Cf[-nb:].reshape(nb, -1)
#%%
def select_masks(ycr, shape):
    m = to_3D(ycr,shape=shape, order='F')
    frame = m[:10000].std(axis=0)
    plt.imshow(frame, cmap=mpl_cm.Greys_r)
    pts = []
    while not len(pts):
        pts = plt.ginput(0)
        plt.close()
        path = mpl_path.Path(pts)
        mask = np.ones(np.shape(frame), dtype=bool)
        for ridx, row in enumerate(mask):
            for cidx, pt in enumerate(row):
                if path.contains_point([cidx, ridx]):
                    mask[ridx, cidx] = False

    ycr = cm.movie((1.0 - mask)*m).to_2D() 
    plt.plot(((m * (1.0 - mask)).mean(axis=(1, 2))))
    return mask, ycr 
#%%
def signal_filter(sg, fr, freq=1/3, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 3)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg
#%%
def combine_datasets(fnames, num_frames, x_shifts=[3,-3], y_shifts=[3,-3], weights=None, shape=(15,15)):
    mm = 0
    ephs = []
    times_e = []
    times_v = []
    volt = []
    if weights is None:
        weights = [1/len(fnames)]*len(fnames)
    for name,x_shift, y_shift, weight in zip(fnames,x_shifts,y_shifts, weights):
        new_mov = cm.load(name)
        if new_mov.shape[1] != shape[0]:
            new_mov = new_mov.resize(shape[0]/new_mov.shape[1],shape[1]/new_mov.shape[2],1)
            
        mm += np.roll(new_mov[:num_frames]*weight, (x_shift, y_shift), axis=(1,2))
    
        name_traces = '/'.join(name.split('/')[:-2] + ['data_new', name.split('/')[-1][:-4]+'_output.npz'])
        #%
        try:
            with np.load(name_traces, allow_pickle=True) as ld:
                dict1 = dict(ld)
                time_v = dict1['v_t'] - dict1['v_t'][0]
                time_e = dict1['e_t'] - dict1['e_t'][0]
                time_v = time_v[:num_frames]
                eph = normalize(dict1['e_sg'][time_e<time_v[-1]])
                time_e = time_e[time_e<time_v[-1]]            
                time_e /= np.max(time_e)
                time_v /= np.max(time_v)
                trep = normalize(dict1['v_sg'][:num_frames])
                times_e.append(time_e)
                times_v.append(time_v)
                ephs.append(eph)
                volt.append(trep)
        except:
            volt, ephs, times_v, times_e = [None]*4
            
        #%
        
    return mm, volt, ephs, times_v, times_e
#%%
c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
#%%
def normalize(ss):
    aa = (ss-np.percentile(ss,1, axis=0))/(np.percentile(ss,99, axis=0)-np.percentile(ss,1, axis=0))
    aa -= np.median(aa)
#    aa /= estimate_running_std(aa)
    return aa
#%%
#m1 = cm.load('403106_3min_rois_mc_lp.hdf5') 
##%%#
#mcr = cm.load('/home/andrea/NEL-LAB Dropbox/Andrea Giovannucci/Kaspar-Andrea/exampledata/Other/403106_3min_rois_mc_lp_crop.hdf5')
##%%
#mcr = cm.load('/home/andrea/NEL-LAB Dropbox/Andrea Giovannucci/Kaspar-Andrea/exampledata/Other/403106_3min_rois_mc_lp_crop_2.hdf5')
##%%
#fname = '/home/nel/data/voltage_data/Marton/454597/Cell_0/40x_patch1/movie/40x_patch1_000_mc_small.hdf5'
#%%
fnames = ['/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/454597_Cell_0_40x_patch1.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/462149_Cell_1_40x_1xtube_10A1.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/462149_Cell_1_40x_1xtube_10A2.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/456462_Cell_3_40x_1xtube_10A2.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/456462_Cell_3_40x_1xtube_10A3.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/456462_Cell_5_40x_1xtube_10A5.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/456462_Cell_5_40x_1xtube_10A6.tif',
'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/456462_Cell_5_40x_1xtube_10A7.tif']
#'/Users/agiovann/NEL-LAB Dropbox/NEL/Papers/VolPy/Marton/video_small_region/462149_Cell_1_40x_1xtube_10A1_10A2.tif']
x_shifts = [3,-3]
y_shifts = [3,-3]
name_set = fnames[1:3]
#x_shifts = [0]
#y_shifts = [0]
#name_set = fnames[1:2]
num_frames = 80000
frate = 400
all_coeffs = []
for idx in range(1):#fnames:    
    
    mcr_orig, volt, ephs, times_v, times_e = combine_datasets(name_set, num_frames,
                                                              x_shifts=x_shifts, 
                                                              y_shifts=y_shifts, 
                                                              weights=None)
         
    mcr_mc = mcr_orig
    mcr = -mcr_mc   
    ycr_orig = mcr.to_2D()
    ycr = mcr.to_2D() 
    ycr_filt = signal_filter(ycr.T,freq = 1/3, fr=frate).T
    ycr = ycr_filt
    plt.figure()
    plt.plot(times_v[0],ycr.mean(axis=(1)))
##%%
#mcr_lc = local_correlations_movie_offline(name_set[0], window=50, stride=20, dview=dview, Tot_frames=10000)
#ycr_lc = mcr_lc.to_2D()
#%%
#    n_components = 1
#    model = NMF(n_components=n_components, init='nndsvda', max_iter=200, verbose=True)
#    W = model.fit_transform(np.maximum(ycr,0))
#    H = model.components_
#    plt.figure();plt.plot(W + np.arange(n_components)/1000) 
#    for i in range(n_components):
#        plt.figure();plt.imshow(H[i].reshape(mcr.shape[1:], order='F'), cmap='gray') 
#        
    #%%
    n_comps = len(x_shifts)
#    n_comps = 2
    num_frames = 20000
    y_now = ycr[:num_frames,:].copy()
    W_tot = []
    H_tot = []
    for i in range(n_comps):
        n_components = 1        
        model = NMF(n_components=n_components, init='nndsvd', max_iter=1000, verbose=True)
        plt.figure()
        mask, y_use = select_masks(y_now, mcr[:num_frames].shape)
        #model = PCA(n_components=n_components)
        W = model.fit_transform(np.maximum(y_use,0))#, H=H, W=W)
        H = model.components_
        W_tot.append(W)
        H_tot.append(H)
        plt.figure();plt.plot(W);
        y_now = y_now - W@H
        for i in range(n_components):
            plt.figure();plt.imshow(H[i].reshape(mcr.shape[1:], order='F'), cmap='gray')
            
    H = np.vstack(H_tot)
    W = np.hstack(W_tot)
    #%%
    fe = slice(0,None)
    from scipy.optimize import nnls
    Cf_pref = np.array([nnls(H.T,y)[0] for y in (ycr-ycr.min())[fe]])
    trr_pref = Cf_pref[:,:]
    trr_pref -= np.min(trr_pref, axis=0)
    Cf_postf = np.array([nnls(H.T,y)[0] for y in -ycr_orig[fe]])
    trr_postf = signal_filter(-Cf_postf.T,freq = 1/3, fr=frate).T
    trr_postf -= np.min(trr_postf, axis=0)
    #%%
    plt.plot(trr_postf)
    plt.plot(trr_pref+0.1)
    
    #%%
    count = 0
    for trep, eph, time_v, time_e in list(zip(volt, ephs, times_v, times_e)):
        plt.plot(time_v[fe],normalize((trep)), label='vpy ' + str(count))
        plt.plot(time_e, normalize(eph)/5-2, label='eph ' + str(count))
        all_coeffs.append(np.corrcoef(trep,trr.T)[1:,0])
        idx_match = np.argmax(all_coeffs[-1])
        plt.plot(time_v[fe],normalize((trr[:, idx_match])), label='online ' + str(count)) 
        count+=1

    plt.legend()        
    
   
#    print([np.corrcoef(ccff,trep)[0,1] for ccff in Cf.T])
#    idx_tr = np.argmax([np.corrcoef(ccff,trep)[0,1] for ccff in Cf.T])
 
#%%
n_components = 2
model = NMF(n_components=n_components, init='custom', max_iter=10, verbose=True) 
W = model.fit_transform(np.maximum(ycr,0), H=H, W=W)
H = model.components_  
for i in range(n_components):
        plt.figure();plt.imshow(H[i].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
cm.movie(to_3D(y_now,shape=mcr.shape, order='F')).resize(1,1,.1).play(fr=40, magnification=4)

#%%
tr_lc, D_lc = spams.nmf(np.asfortranarray(ycr), K=1, return_lasso=True)   
plt.figure();plt.plot(tr_lc) 
plt.figure();plt.imshow(D_lc.T.toarray()[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc.T.toarray()[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc.T.toarray()[:,2].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
D_lc,tr_lc = spams.nmf(np.asfortranarray(ycr.T), K=2, return_lasso=True, )   
plt.figure();plt.plot(tr_lc.T.toarray()) 
plt.figure();plt.imshow(D_lc[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D_lc[:,2].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
(D,model) = spams.trainDL(np.asfortranarray(np.diff(ycr, axis=0).T), K=2, lambda1=10, return_model=True)
plt.figure();plt.imshow(D[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray')
plt.figure();plt.imshow(D[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray')
#plt.figure();plt.imshow(D[:,2].reshape(mcr[1:].shape[1:], order='F'), cmap='gray')

#%%
D,tr = spams.nnsc(np.asfortranarray(ycr.T), K=2, return_lasso=True, lambda1=0)
#%%
D,tr = spams.nmf(np.asfortranarray(np.abs(ycr.T)), K=2, return_lasso=True) 
#%%
D,tr = spams.nnsc(np.asfortranarray(ycr.T), K=2, return_lasso=True, lambda1=1)
#%%
plt.figure();plt.plot(tr.T.toarray()) 
plt.figure();plt.imshow(D[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray');plt.title('comp1') 
plt.figure();plt.imshow(D[:,0].reshape(mcr.shape[1:], order='F'), cmap='gray');plt.title('comp0')

#%%
plt.figure();plt.plot(tr.T.toarray()) 
plt.figure();plt.imshow(D[:,1].reshape(mcr.shape[1:], order='F'), cmap='gray') 
#%%
idx = np.round((dict1['v_sp']-dict1['v_t'][0])/np.median(np.diff(dict1['v_t']))).astype(np.int)
plt.figure(); plt.imshow(np.median(mcr_lc[idx[:1000]+3], axis=0))
#%%
plt.figure();plt.plot(dict1['v_t'],W)
plt.plot(dict1['e_sp'],[1]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[2]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[3]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[4]*len(dict1['e_sp']),'k|')
plt.plot(dict1['e_sp'],[5]*len(dict1['e_sp']),'k|')
plt.plot(dict1['v_t'], dict1['v_sg']*100)
#%%
spikes = ((dict1['e_sp']-dict1['e_t'][0])/np.mean(np.diff(dict1['v_t']))).astype(np.int)+1

#%%
plt.close()
fe = slice(0,None)
from scipy.optimize import nnls
Cf = np.array([nnls(H.T,y)[0] for y in (ycr)[fe]])
trr = Cf[:,:]
plt.plot(dict1['v_t'][fe],(trr-np.min(trr, axis=0))/(np.max(trr, axis=0)-np.min(trr, axis=0)))
trep = dict1['v_sg'][fe]
plt.plot(dict1['v_t'][fe], 1+(trep-np.min(trep))/(np.max(trep)-np.min(trep)),'c')
eph = dict1['e_sg']
plt.plot(dict1['e_t'], normalize(eph)-1,'k')
#plt.plot(dict1['e_sp'],[1]*len(dict1['e_sp']),'k|')
#%%
cm.movie(to_3D((H[[0,1],:].T@Cf[:,[0,1]].T).T,shape=mcr.shape, order='F')).play(magnification=4)
#%%
cm.movie(to_3D((ycr.T-H[[0],:].T@Cf[:,[0]].T).T,shape=mcr.shape, order='F')).play(magnification=4)
#%%
#from caiman.source_extraction.cnmf.initialization import hals
#A = []
#C = []
#for fname in np.array(fnames)[[1,2]]:
#    print(fname)
#    try:
#        name_traces = '/'.join(fname.split('/')[:-2] + ['data_new', fname.split('/')[-1][:-4]+'_output.npz'])
#        #%
#        with np.load(name_traces, allow_pickle=True) as ld:
#            dict1 = dict(ld)
#        #%
#    except:
#        print('failed')
#        
#
#    A.append(cm.movie(to_3D(ycr_filt[spikes,:],shape=mcr[spikes].shape, order='F')).zproject()[None,:,:].to_2D().T)
#    C.append(((ycr+8)@A[-1]).T)
#
#A =np.hstack(A) 
#C =np.vstack(C)
#
#H,Cf,b,f = hals((ycr+8).T, A, C, np.zeros_like(A), np.zeros_like(C))
#H=H.T
#Cf=Cf.T
#%%
#plt.close()
fe = slice(0,None)
from scipy.optimize import nnls
Cf = np.array([nnls(H.T,y)[0] for y in (ycr)[fe]])
#%
for fname in np.array(fnames)[[1,2]]:
    try:
        name_traces = '/'.join(fname.split('/')[:-2] + ['data_new', fname.split('/')[-1][:-4]+'_output.npz'])
        #%
        with np.load(name_traces, allow_pickle=True) as ld:
            dict1 = dict(ld)
            
        print(fname)

        #%
    except:
        print('failed')
        
    time_v = dict1['v_t'] - dict1['v_t'][0]
    time_e = dict1['e_t'] - dict1['e_t'][0]
    time_e /= np.max(time_e)
    time_v /= np.max(time_v)
    trep = dict1['v_sg'][fe]
    plt.plot(time_v[fe], 1+normalize(trep))
    eph = dict1['e_sg']
    plt.plot(time_e, normalize(eph)-1)
    print([np.corrcoef(ccff,trep)[0,1] for ccff in Cf.T])
    idx_tr = np.argmax([np.corrcoef(ccff,trep)[0,1] for ccff in Cf.T])
    trr = Cf[:,idx_tr]
    plt.plot(time_v[fe],(trr-np.min(trr, axis=0))/(np.max(trr, axis=0)-np.min(trr, axis=0)))  
#%%
H1,Cf1,b1,f1 = hals(cm.movie(to_3D(ycr,shape=(mcr.shape[0],15,15), order='F')).transpose([1,2,0]), H.T, Cf.T, np.zeros((ycr.shape[1],1)), np.zeros((1,ycr.shape[0])), bSiz=10, maxIter=10)
H1=H1.T
Cf1=Cf1.T  
for i in range(3):
        plt.figure();plt.imshow(H1[i].reshape(mcr.shape[1:], order='F'), cmap='gray')
#%%
fun = lambda x: normalize(np.diff(x))   
fun = lambda x: normalize(x)        
     
#plt.plot(normalize(fun(Cf1[:,1])),'c')
#plt.plot(normalize(fun(trr[:, idx_match])), label='online ' + str(count)) 
plt.plot(time_v[:], normalize(fun(trep)), label='vpy ' + str(count))
#plt.plot(time_e[1:],normalize(fun(eph))+2, label='vpy ' + str(count))
plt.plot(time_v[:], normalize(fun(Cf1[:,0])))
