#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:02:16 2020
This file is for timing of spike detection algorithm
@author: caichangjia
"""

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({'pdf.fonttype' : 42, 
                     'ps.fonttype' : 42, 
                     'legend.frameon' : False, 
                     'axes.spines.right' :  False, 
                     'axes.spines.top' : False})
import numpy as np
import pyximport
pyximport.install()
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
from fiola.utilities import signal_filter


#%%

