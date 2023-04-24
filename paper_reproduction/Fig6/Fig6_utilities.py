#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:56:17 2022

@author: nel
"""
from caiman.base.rois import com
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler

def remove_neurons_with_sparse_activities(trace, do_remove=True, std_level=5, timepoints=10):
    from sklearn.preprocessing import StandardScaler
    print(f'original trace shape {trace.shape}')
    trace_s = StandardScaler().fit_transform(trace)
    import pdb
    pdb.set_trace()
    if do_remove:
        select = []
        high_act = []
        for idx in range(len(trace_s.T)):
            t = trace_s[:, idx]
            select.append(len(t[t>t.std() * std_level]) > timepoints)
            high_act.append(len(t[t>t.std() * std_level]))
        #plt.plot(trace_s[:, ~np.array(select)][:, 15])
        trace_s = trace_s[:, select]
        high_act = np.array(high_act)[select]
        sort = np.argsort(high_act)[::-1]
        trace_s = trace_s[:, sort]
    print(f'after removing neurons trace shape {trace_s.shape}')
    return trace_s, select

def sort_neurons(trace, std_level=5, timepoints=10):
    from sklearn.preprocessing import StandardScaler
    print(f'original trace shape {trace.shape}')
    trace_s = StandardScaler().fit_transform(trace)
    high_act = []
    for idx in range(len(trace_s.T)):
        t = trace_s[:, idx]
        high_act.append(len(t[t>t.std() * std_level]))
    sort = np.argsort(high_act)[::-1]
    return sort

def signal_filter(sg, freq, fr, order=3, mode='high'):
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
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    #sg = np.single(signal.filtfilt(b, a, sg))
    return sg

def select_neurons_within_regions(sp_raw, y_limit):
    selection = {}
    for key, values in sp_raw.items():
        if 'Suite2p' in key:    
            centers = np.array([values[i]['med'] for i in range(len(values))])
            select = np.where(np.logical_and(centers[:, 0] > y_limit[0], centers[:, 0] <= y_limit[1]))[0]
        
        else:
            centers = com(values.estimates.A, 796, 512)
            select = np.where(np.logical_and(centers[:, 0] > y_limit[0], centers[:, 0] <= y_limit[1]))[0]
        selection[key] = select
    [print(s.shape) for s in selection.values()]
    return selection

def run_spatial_preprocess(sp_raw, selection):
    sp_processed = {}
    for key, values in sp_raw.items():
        if 'Suite2p' in key:   
            masks = np.zeros((796, 512, len(values)))
            for i, stat in enumerate(values):
                masks[stat['ypix'], stat['xpix'], i] = 1
            A_2p = scipy.sparse.csc_matrix(
                np.reshape(masks[:, :, :], (np.prod([796, 512]), -1), order='F'))
            spatial = A_2p
        else:
            spatial = values.estimates.A
        sp_processed[key] = spatial[:, selection[key]]
    [print(s.shape) for s in sp_processed.values()]
    return sp_processed
    
def run_trace_preprocess(t_raw, selection, sigma=12):
    if selection is not None:
        print('removing neurons')
        t_rm = {}
        for key in t_raw.keys():
            t_rm[key] = t_raw[key][:, selection[key]]
    else:
        t_rm = t_raw
        print('not removing neurons')
    [print(tt.shape) for tt in t_rm.values()]
    
    print('normalizing')
    t_rs = {}
    for key in t_rm.keys():
        t_rs[key] = StandardScaler().fit_transform(t_rm[key])
    [print(tt.shape) for tt in t_rs.values()]

    print(f'gaussian filtering with ssd {sigma}')
    t_g = {}
    for key in t_rs.keys():
        t_g[key] = t_rs[key].copy()
        for i in range(t_g[key].shape[1]):
            t_g[key][:, i] = gaussian_filter1d(t_g[key][:, i], sigma=sigma)
    [print(key, tt.shape) for key, tt in t_g.items()]
    return t_g, t_rs, t_rm

def cross_validation_regularizer_strength(x, y, normalize, n_splits=None, alpha_list=None):
    score = {}
    kf = KFold(n_splits=n_splits)
    for alpha in alpha_list:
        s = []
        for train, test in kf.split(x):
            if normalize:
                x_train = StandardScaler().fit_transform(x[train])
                y_train = StandardScaler().fit_transform(y[train][:, None])[:, 0]        
                x_test = StandardScaler().fit_transform(x[test])
                y_test = StandardScaler().fit_transform(y[test][:, None])[:, 0]        
            else:
                x_train = x[train]
                y_train = y[train]
                x_test = x[test]
                y_test = y[test]
            clf = Ridge(alpha=alpha)
            clf.fit(x_train, y_train)  
            s.append(clf.score(x_test, y_test))
        score[alpha] = np.mean(s)
    alpha_best = max(score, key=score.get)
    print(f'best alpha: {alpha_best}')
    return alpha_best

def cross_validation_ridge(X, Y, normalize=True, n_splits=None, alpha_list=None):
    score = []
    predict = []
    kf = KFold(n_splits=n_splits)
    print(f'normalize:{normalize}')
    for idx, [cv_train, cv_test] in enumerate(kf.split(X)):
        print(f'processing fold {idx}')
        alpha_best = cross_validation_regularizer_strength(x=X[cv_train], y=Y[cv_train], normalize=normalize, n_splits=n_splits, alpha_list=alpha_list)
        if normalize:
            X_train = StandardScaler().fit_transform(X[cv_train])
            Y_train = StandardScaler().fit_transform(Y[cv_train][:, None])[:, 0]        
            X_test = StandardScaler().fit_transform(X[cv_test])
            Y_test = StandardScaler().fit_transform(Y[cv_test][:, None])[:, 0]        
        else:
            X_train = X[cv_train]
            Y_train = Y[cv_train]
            X_test = X[cv_test]
            Y_test = Y[cv_test]
        clf = Ridge(alpha=alpha_best)
        clf.fit(X_train, Y_train)  
        score.append(clf.score(X_test, Y_test))
        predict.append({'cv_test':cv_test, 'Y':Y_test, 'Y_pr':clf.predict(X_test)})
    return score, predict

def masks_to_binary(masks):
    binary_masks = []
    for idx, mask in enumerate(masks):
        if idx % 100 == 0:
            print(f'{idx} mask processed')
        mask[mask>np.percentile(mask, 99.98)] = 1
        mask[mask!=1]=0
        binary_masks.append(mask)
    binary_masks = np.array(binary_masks)
    return binary_masks

def view_compare_components(t1, t2, A, img, area=1):
    """ View spatial and temporal components interactively
    Args:
        estimates: dict
            estimates dictionary contain results of VolPy

        idx: list
            index of selected neurons
    """
    fig = plt.figure(figsize=(10, 10))
    dims = img.shape
    centers = com(A, dims[0], dims[1])
    spatial = A.T.reshape([-1, dims[0], dims[1]], order='F')
    
    xx = dims[1]
    yy = dims[0]
    yl = int(796 * np.sqrt(area))
    xl = int(512 * np.sqrt(area))
    x_range = [(xx - xl) // 2, (xx - xl) // 2 + xl ]
    y_range = [(yy - yl) // 2, (yy - yl) // 2 + yl ]
    
    select1 = np.where(np.logical_and(centers[:, 1] >= x_range[0], centers[:, 1] <= x_range[1]))[0]
    select2 = np.where(np.logical_and(centers[:, 0] >= y_range[0], centers[:, 0] <= y_range[1]))[0]
    select = np.intersect1d(select1, select2)    
    
    n = len(select)
    print(n)
    t1 = t1[select]
    t2 = t2[select]
    spatial = spatial[select, y_range[0]:y_range[1], x_range[0]:x_range[1]]
    img = img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

    axcomp = plt.axes([0.05, 0.05, 0.9, 0.03])
    ax1 = plt.axes([0.05, 0.55, 0.4, 0.4])
    ax3 = plt.axes([0.55, 0.55, 0.4, 0.4])
    ax2 = plt.axes([0.05, 0.1, 0.9, 0.4])    
    s_comp = Slider(axcomp, 'Component', 0, n, valinit=0)
    vmax = np.percentile(img, 98)
    
    print('!!!')
    
    def arrow_key_image_control(event):

        if event.key == 'left':
            new_val = np.round(s_comp.val - 1)
            if new_val < 0:
                new_val = 0
            s_comp.set_val(new_val)

        elif event.key == 'right':
            new_val = np.round(s_comp.val + 1)
            if new_val > n :
                new_val = n  
            s_comp.set_val(new_val)
        
    def update(val):
        i = np.int(np.round(s_comp.val))
        print(f'component:{i}')

        if i < n:
            
            ax1.cla()
            imgtmp = spatial[i]
            ax1.imshow(imgtmp, interpolation='None', cmap=plt.cm.gray, vmax=np.max(imgtmp)*0.5, vmin=0)
            ax1.set_title(f'Spatial component {i+1}')
            ax1.axis('off')
            
            ax2.cla()
            ax2.plot(t1[i], alpha=0.8, label='fiola traces')
            ax2.plot(t2[i], alpha=0.8, label='meanroi traces')
            ax2.legend()
            ax2.set_title(f'Signal {i+1}')
            
            ax3.cla()
            ax3.imshow(img, interpolation='None', cmap=plt.cm.gray, vmax=vmax)
            imgtmp2 = imgtmp.copy()
            imgtmp2[imgtmp2 == 0] = np.nan
            ax3.imshow(imgtmp2, interpolation='None',
                       alpha=0.5, cmap=plt.cm.hot)
            ax3.axis('off')
            
    s_comp.on_changed(update)
    s_comp.set_val(0)
    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    plt.show()
    
def barplot_annotate_brackets(data, center1, center2, height, dy, yerr=None):
    """ 
    Annotate barplot with p-values.
    :param data: string to write or number for generating asterixes
    :param center1: centers of first bar (like plt.bar() input)
    :param center2: centers of second bar (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    """

    if type(data) is str:
        text = data
    else:
        if data > 0.05:
            text = 'ns'
        elif data <=0.001:
            text = '***'
        elif data <= 0.01:
            text = '**'
        elif data <= 0.05:
            text = '*'

    lx, ly = center1, height
    rx, ry = center2, height
    if yerr:
        ly += yerr
        ry += yerr

    y = max(ly, ry)
    barx = [lx, rx]
    bary = [y, y]
    mid = ((lx+rx)/2, y)
    plt.plot(barx, bary, c='black')
    plt.plot([lx, lx], [y, y-dy], c='black')
    plt.plot([rx, rx], [y, y-dy], c='black')
    kwargs = dict(ha='center', va='bottom')
    plt.text(*mid, text, **kwargs)


from scipy.stats import wilcoxon, ttest_rel, ttest_ind
def barplot_pvalue(r_all, methods, colors, ax, dev=0.1, width=0.2, capsize=0, alpha=1):
    """
    Plot bar plot with p_values and significance
    """
    p_value = {}
    #labels = ['FIOLA', 'VolPy', 'MeanROI']
    labels = methods
    #labels = methods.copy()
    #labels[-1] = 'CaImAn_Batch'
    for batch in range(len(r_all)):
        rr = r_all[batch]
        
        if len(methods) % 2 == 0:
            shift = len(methods) // 2 - 1 / 2
        else:
            shift = len(methods) // 2
    
        for j, method in enumerate(methods):
            ax.scatter(rand_jitter([batch + width * (j - shift)] * len(rr[method]), dev=dev), rr[method], color='black', alpha=alpha, s=15, zorder=2, facecolor='none')
            ax.bar(batch + width * (j - shift), [np.mean(rr[method])], width=width,
            yerr=[[np.std(rr[method])], [np.std(rr[method])]],
            color=colors[j], label=f'{method}', alpha=1, capsize=capsize)

    
            for k in range(j+1, len(methods)):
                dat = ttest_rel(rr[methods[j]], rr[methods[k]], alternative='two-sided').pvalue 
                #dat = ttest_ind(rr[methods[j]], rr[methods[k]], alternative='two-sided').pvalue 
                
                if dat <= 0.05:
                    print(f'batch: {batch}; {methods[j]}, {methods[k]}; {np.round(dat, 4)}')
                    print(f'batch: {batch}; {methods[j]}, {methods[k]}; {np.round(dat, 10)}')

                    barplot_annotate_brackets(dat, batch + width * (j - shift), batch + width * (k - shift), 
                                              height = 0.1 * (j + k)/2 + np.array(list(rr.values())).max(), 
                                              dy=0.003)
                if f'{labels[j]} vs {labels[k]}' in p_value:
                    p_value[f'{labels[j]} vs {labels[k]}'].append(float(f'{dat:.2e}'))
                else:
                    p_value[f'{labels[j]} vs {labels[k]}'] = []
                    p_value[f'{labels[j]} vs {labels[k]}'].append(float(f'{dat:.2e}'))
    return p_value
def rand_jitter(arr, dev=10):    
    return arr + np.random.randn(len(arr)) * dev        