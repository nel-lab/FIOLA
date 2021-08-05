#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:30:42 2020
Parameters for online analysis of fluorescence (calcium/voltage) imaging data.
@author: @agiovann, @caichangjia, @cynthia
"""

import logging
import numpy as np

class fiolaparams(object):
    def __init__(self, fnames=None, fr=None, ROIs=None, mode='voltage', num_frames_init=10000, num_frames_total=20000, 
                 ms=[10,10], offline_mc_batch_size=200, border_to_0=0, freq_detrend = 1/3, do_plot_init=True, erosion=0, 
                 hals_movie='hp_thresh', use_rank_one_nmf=True, semi_nmf=False,
                 update_bg=False, use_spikes=False, use_batch=True, batch_size=1, 
                 center_dims=None, initialize_with_gpu=False, 
                 window = 10000, step = 5000, detrend=True, flip=True, 
                 do_scale=False, template_window=2, robust_std=False, freq=15,adaptive_threshold=True, 
                 thresh_range=[3.5, 5], minimal_thresh=3.0, mfp=0.2, online_filter_method = 'median_filter',
                 filt_window = 15, do_plot=False, params_dict={}):
        """Class for setting parameters for online fluorescece imaging analysis. Including parameters for the data, motion correction and
        spike detection. The prefered way to set parameters is by using the set function, where a subclass is determined
        and a dictionary is passed. The whole dictionary can also be initialized at once by passing a dictionary
        params_dict when initializing the fiolaparams object.
        """
        self.data = {
            'fnames': fnames, # name of the movie
            'fr': fr, # sample rate of the movie
            'ROIs': ROIs, # a 3-d matrix contains all region of interests
            'mode': mode, # 'voltage' or 'calcium 'fluorescence indicator
            'num_frames_init': num_frames_init, # number of frames used for initialization
            'num_frames_total':num_frames_total # estimated total number of frames for processing, this is used for generating matrix to store data            
        }

        self.mc_nnls = {
            'ms':ms, # maximum shift in x and y axis respectively. Will not perform motion correction if None.
            'offline_mc_batch_size': offline_mc_batch_size, # number of frames for one batch to perform offline motion correction
            'border_to_0': border_to_0,  # border of the movie will copy signals from the nearby pixels
            'freq_detrend': freq_detrend, # high-pass frequency for removing baseline, used for init of spatial footprint
            'do_plot_init': do_plot_init, # plot the spatial mask result for init of spaital footprint
            'erosion': erosion, # number of pixels to erode the input masks before performing rank-1 NMF
            'hals_movie': hals_movie, # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); original movie (orig)
            'use_rank_one_nmf': use_rank_one_nmf, # whether to use rank-1 nmf, if False the algorithm will use initial masks and average signals as initialization for the HALS
            'semi_nmf': semi_nmf, # whether use semi-nmf (with no constraint in temporal components) instead of normal NMF
            'update_bg': update_bg, # update background components for spatial footprints
            'use_spikes': use_spikes, # whether to use reconstructed signals for the HALS algorithm
            'use_batch':use_batch, # whether to process a batch of frames (greater or equal to 1) at the same time. Process one frame a time if False 
            'batch_size':batch_size, # number of frames processing at the same time using gpu 
            'center_dims':center_dims, # template dimensions for motion correction. If None, the input will the the shape of the FOV
            'initialize_with_gpu': initialize_with_gpu # whether to use gpu for performing nnls during initialization 
        }

        self.spike = {
            'window': window, # window for updating statistics
            'step': step, # step for updating statistics
            'flip': flip, # whether to flip signal to find spikes    
            'detrend': detrend, # whether to remove photobleaching
            'do_scale': do_scale, # whether to scale the input trace or not
            'template_window':template_window, # half window size of the template; will not perform template matching if window size equals 0
            'robust_std':robust_std, # whether to use robust way to estimate noise
            'freq': freq, # frequency for removing subthreshold activity
            'adaptive_threshold': adaptive_threshold, #whether to use adaptive threshold method for deciding threshold level
            'thresh_range':thresh_range, # range of threshold factor. Real threshold is threshold factor multiply by the estimated noise level
            'minimal_thresh':minimal_thresh, # minimal of the threshold 
            'mfp': mfp, #  Maximum estimated false positive. An upper bound for estimated false positive rate based on noise
            'online_filter_method': online_filter_method,
            'filt_window': filt_window, # window size for removing the subthreshold activities 
            'do_plot': do_plot # Whether to plot or not
        }

        self.change_params(params_dict)

    def set(self, group, val_dict, set_if_not_exists=False, verbose=False):
        """ Add key-value pairs to a group. Existing key-value pairs will be overwritten
            if specified in val_dict, but not deleted.

        Args:
            group: The name of the group.
            val_dict: A dictionary with key-value pairs to be set for the group.
            set_if_not_exists: Whether to set a key-value pair in a group if the key does not currently exist in the group.
        """

        if not hasattr(self, group):
            raise KeyError('No group in CNMFParams named {0}'.format(group))

        d = getattr(self, group)
        for k, v in val_dict.items():
            if k not in d and not set_if_not_exists:
                if verbose:
                    logging.warning(
                        "NOT setting value of key {0} in group {1}, because no prior key existed...".format(k, group))
            else:
                if np.any(d[k] != v):
                    logging.warning(
                        "Changing key {0} in group {1} from {2} to {3}".format(k, group, d[k], v))
                d[k] = v

    def get(self, group, key):
        """ Get a value for a given group and key. Raises an exception if no such group/key combination exists.

        Args:
            group: The name of the group.
            key: The key for the property in the group of interest.

        Returns: The value for the group/key combination.
        """

        if not hasattr(self, group):
            raise KeyError('No group in CNMFParams named {0}'.format(group))

        d = getattr(self, group)
        if key not in d:
            raise KeyError('No key {0} in group {1}'.format(key, group))

        return d[key]

    def get_group(self, group):
        """ Get the dictionary of key-value pairs for a group.

        Args:
            group: The name of the group.
        """

        if not hasattr(self, group):
            raise KeyError('No group in CNMFParams named {0}'.format(group))

        return getattr(self, group)

    def change_params(self, params_dict, verbose=False):
        for gr in list(self.__dict__.keys()):
            self.set(gr, params_dict, verbose=verbose)
        for k, v in params_dict.items():
            flag = True
            for gr in list(self.__dict__.keys()):
                d = getattr(self, gr)
                if k in d:
                    flag = False
            if flag:
                logging.warning('No parameter {0} found!'.format(k))
        return self
