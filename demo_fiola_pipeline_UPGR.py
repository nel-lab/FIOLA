#!/usr/bin/env python

"""
Complete demo pipeline for processing two photon calcium imaging data using the
CaImAn batch algorithm. The processing pipeline included motion correction,
source extraction and deconvolution. The demo shows how to construct the
params, MotionCorrect and cnmf objects and call the relevant functions. You
can also run a large part of the pipeline with a single method (cnmf.fit_file)
See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline.ipynb)
Dataset couresy of Sue Ann Koay and David Tank (Princeton University)

This demo pertains to two photon data. For a complete analysis pipeline for
one photon microendoscopic data see demo_pipeline_cnmfE.py

copyright GNU General Public License v2.0
authors: @agiovann and @epnev
"""

import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass


import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline
from caiman.source_extraction.cnmf.utilities import get_file_size


logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
#%%    
c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

#%% dataset dependent parameters
display_images = False

fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/mov_R2_20190219T210000_3000.hdf5']  # filename to be processed
# fnames = ['/home/nel/caiman_data/example_movies/demoMovie.tif']  # filename to be processed
# fnames = ['/home/nel/caiman_data/example_movies/Sue_2x_3000_40_-46.tif']
num_frames_total = get_file_size(fnames)[-1]         # nummbber of total frames including initialization
num_frames_init = num_frames_total//2
fr = 30             # imaging rate in frames per second
decay_time = 0.4    # length of a typical transient in seconds
dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
# note the lower than usual spatial resolution here
max_shift_um = (12., 12.)       # maximum shift in um
patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
# motion correction parameters
pw_rigid = True       # flag to select rigid vs pw_rigid motion correction
# maximum allowed rigid shift in pixels
max_shifts = [int(a/b) for a, b in zip(max_shift_um, dxy)]
# start a new patch for pw-rigid motion correction every x pixels
strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
# overlap between pathes (size of patch in pixels: strides+overlaps)
overlaps = (24, 24)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 3

mc_dict = {
    'fnames': fnames,
    'fr': fr,
    'decay_time': decay_time,
    'dxy': dxy,
    'pw_rigid': pw_rigid,
    'max_shifts': max_shifts,
    'strides': strides,
    'overlaps': overlaps,
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': 'copy',
}

opts = params.CNMFParams(params_dict=mc_dict)
#%% Motion correction and memory mapping
time_init = time()
mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
mc.motion_correct(save_movie=True)
border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0, slices=[slice(0,num_frames_init)])  # exclude borders
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
# %% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%%
f_F_mmap = mc.mmap_file[0]
Cns = local_correlations_movie_offline(f_F_mmap,
                                   remove_baseline=True, window=1000, stride=1000,
                                   winSize_baseline=100, quantil_min_baseline=10,
                                   dview=dview)
Cn = Cns.max(axis=0)
Cn[np.isnan(Cn)] = 0
if display_images: 
   
    plt.imshow(Cn,vmax=0.5)
# %%  parameters for source extraction and deconvolution
p = 1                    # order of the autoregressive system
gnb = 1                  # number of global background components
merge_thr = 0.85         # merging threshold, max correlation allowed
rf = 15
# half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6          # amount of overlap between the patches in pixels
K = 5                    # number of components per patch
gSig = [4, 4]            # expected half size of neurons in pixels
# initialization method (if analyzing dendritic data using 'sparse_nmf')
method_init = 'greedy_roi'
ssub = 2                     # spatial subsampling during initialization
tsub = 2                     # temporal subsampling during intialization

# parameters for component evaluation
opts_dict = {'fnames': fnames,
             'p': p,
             'fr': fr,
             'nb': gnb,
             'rf': rf,
             'K': K,
             'gSig': gSig,
             'stride': stride_cnmf,
             'method_init': method_init,
             'rolling_sum': True,
             'merge_thr': merge_thr,
             'n_processes': n_processes,
             'only_init': True,
             'ssub': ssub,
             'tsub': tsub}

opts.change_params(params_dict=opts_dict);
# %% RUN CNMF ON PATCHES
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)
# %% COMPONENT EVALUATION
min_SNR = 1.0  # signal to noise ratio for accepting a component
rval_thr = 0.75  # space correlation threshold for accepting a component
cnn_thr = 0.3  # threshold for CNN based classifier
cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected

cnm.params.set('quality', {'decay_time': decay_time,
                       'min_SNR': min_SNR,
                       'rval_thr': rval_thr,
                       'use_cnn': False,
                       'min_cnn_thr': cnn_thr,
                       'cnn_lowest': cnn_lowest})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
print(len(cnm.estimates.idx_components))
time_patch = time()
#%%
if display_images:
    cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)
#%%
cnm.estimates.select_components(use_object=True)
# %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
cnm2 = cnm.refit(images, dview=dview)
time_end = time() 
print(time_end- time_init)
# %% COMPONENT EVALUATION
min_SNR = 1.1  # signal to noise ratio for accepting a component
rval_thr = 0.85  # space correlation threshold for accepting a component
cnn_thr = 0.15  # threshold for CNN based classifier
cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected

cnm2.params.set('quality', {'decay_time': decay_time,
                           'min_SNR': min_SNR,
                           'rval_thr': rval_thr,
                           'use_cnn': True,
                           'min_cnn_thr': cnn_thr,
                           'cnn_lowest': cnn_lowest})
cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
print(len(cnm2.estimates.idx_components))

# %% PLOT COMPONENTS
cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
# %% VIEW TRACES (accepted and rejected)
if display_images:
    cnm2.estimates.view_components(images, img=Cn,
                                  idx=cnm2.estimates.idx_components)
    cnm2.estimates.view_components(images, img=Cn,
                                  idx=cnm2.estimates.idx_components_bad)
#%% update object with selected components
cnm2.estimates.select_components(use_object=True)
#%% Extract DF/F values
cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
#%% Show final traces
cnm2.estimates.view_components(img=Cn)
#%%
cnm2.mmap_F = f_F_mmap 
cnm2.estimates.Cn = Cn
cnm2.estimates.template = mc.total_template_rig
cnm2.estimates.shifts = mc.shifts_rig
cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')

#%% STOP CLUSTER and clean up log files
cm.stop_server(dview=dview)
log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)

#%%
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import os
from tensorflow.python.client import device_lib
from threading import Thread
from time import time

from fiola.config import load_fiola_config_calcium
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, play, bin_median, to_2D, local_correlations
from caiman.source_extraction.cnmf.utilities import fast_prct_filt as detrend
from caiman.source_extraction.cnmf.utilities import get_file_size

import caiman as cm

from tifffile import memmap, imread

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)
    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

#%%
caiman_file = '/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/memmap__d1_796_d2_512_d3_1_order_C_frames_1500_.hdf5'
fnames = ['/home/nel/NEL-LAB Dropbox/NEL/Papers/VolPy_online/data/dandi_deconding_data/mov_R2_20190219T210000_3000.hdf5']  # filename to be processed

# caiman_file = '/home/nel/caiman_data/example_movies/memmap__d1_60_d2_80_d3_1_order_C_frames_1000_.hdf5'
# fnames = ['/home/nel/caiman_data/example_movies/demoMovie.tif']  # filename to be processed

# caiman_file = '/home/nel/caiman_data/example_movies/memmap__d1_170_d2_170_d3_1_order_C_frames_1500_.hdf5'
# fnames = ['/home/nel/caiman_data/example_movies/Sue_2x_3000_40_-46.tif']

cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
num_frames_total = get_file_size(fnames)[-1]         # nummbber of total frames including initialization
ms = [10, 10]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None. 
estimates = cnm2.estimates
template = cnm2.estimates.template
shift_caiman = cnm2.estimates.shifts
num_frames_init =  cnm2.estimates.C.shape[1]         # number of frames used for initialization
mov = cm.load(fnames,subindices=range(num_frames_init), in_memory=True)
min_mov = mov.min()
#%% Mot corr only
options = load_fiola_config_calcium(fnames, num_frames_total=num_frames_total,
                                    num_frames_init=num_frames_init, 
                                    batch_size=1, ms=ms)

params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
mc_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch_size'], min_mov)
plt.plot(shift_caiman)
plt.plot(shifts_fiola)
#%% NNLS Only
options = load_fiola_config_calcium(fnames, num_frames_total=num_frames_total,
                                    num_frames_init=num_frames_init, 
                                    batch_size=1, ms=ms)

params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
mc_nn_mov = mc_mov#-min_mov
fio.fit_hals(mc_nn_mov, estimates.A, estimates.b)
Ab = fio.Ab
trace_fiola = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch_size']) 
# trace_fiola_no_hals = fio.fit_gpu_nnls(mov, np.hstack((estimates.A.toarray(), estimates.b)), batch_size=fio.params.mc_nnls['offline_batch_size']) 
trace_caiman = np.vstack((estimates.C[:,:num_frames_init] + estimates.YrA[:,:num_frames_init],estimates.f[:,:num_frames_init]))
cc = [np.corrcoef(s1,s2)[0,1] for s1,s2 in zip(detrend(trace_fiola[:-1][cnm2.estimates.SNR_comp>0],30),detrend(trace_caiman[:-1][cnm2.estimates.SNR_comp>0],30))]
plt.hist(cc,30)
plt.figure()
plt.scatter(cnm2.estimates.SNR_comp,cc)
#%% Full Pipeline
options = load_fiola_config_calcium(fnames, num_frames_total=num_frames_total,
                                    num_frames_init=num_frames_init, 
                                    batch_size=1, ms=ms)

params = fiolaparams(params_dict=options)
fio = FIOLA(params=params)
#fio.fit_hals(mc_mov, estimates.A, estimates.b)
fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov)
#%%
# mov_new = cm.load(fnames,subindices=range(num_frames_init, num_frames_total), in_memory=False)
#%%
# memmap_image = memmap(fnames[0])
memmap_image = imread(fnames[0]).astype(np.float32)
time_per_step = np.zeros(num_frames_total-num_frames_init)
traces = np.zeros((num_frames_total-num_frames_init,fio.Ab.shape[-1]), dtype=np.float32)
start = time()
for idx in range(num_frames_init, num_frames_total):
    fio.fit_online_frame(memmap_image[idx:idx+1].astype(np.float32))   # fio.pipeline.saoz.trace[:, i] contains trace at timepoint i        
    traces[idx-num_frames_init] = fio.pipeline.saoz.trace[:,idx-1]
    time_per_step[idx-num_frames_init] = (time()-start)
logging.info(f'total time online: {time()-start}')
logging.info(f'time per frame online: {(time()-start)/(memmap_image.shape[0])}')
plt.plot(np.diff(time_per_step))
#%%
fio.compute_estimates()
fio.view_components(estimates.Cn)
#%%
cnm2.estimates.view_components()
#%%

