from math import exp, log
from numba import njit, prange
import numpy as np

@njit('f4, f4, f4[:], f4[:], f4[:], i4', cache=True)
def fit_next_AR1(y,lg, v,w,l,i):
    v[i], w[i], l[i] = y, 1, 1
    f = exp(lg * l[i-1])
    while (i > 0 and  # backtrack until violations fixed
            v[i-1] / w[i-1] * f > v[i] / w[i]):
        i -= 1
        # merge two pools
        v[i] += v[i+1] * f
        w[i] += w[i+1] * f*f
        l[i] += l[i+1]
        f = exp(lg * l[i-1])
    i += 1
    return v,w,l,i

@njit(['f4[:], f4[:], f4[:,:], f4[:,:], f4[:,:], i4[:]'], parallel=True, cache=True)
def par_fit_next_AR1(y,lg, v,w,l,i):
    N = len(y)
    for k in prange(N):
        v[k],w[k],l[k],i[k] = fit_next_AR1(y[k],lg[k], v[k],w[k],l[k],i[k])

@njit('f4[:], f4, f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], i4[:], i4[:], i4, i4', cache=True)
def fit_next_AR2(y,d,g11,g12,g11g11,g11g12, v,w,t,l,i,tt):
    # [first value, last value, start time, length] of pool
    v[i], w[i], t[i], l[i] = y[tt], y[tt], tt, 1
    ld = log(d)
    while (i > 0 and  # backtrack until violations fixed
            ((((g11[l[i-1]] * v[i-1] + g12[l[i-1]] * w[i-2]) if l[i-1] < 1000
            else exp(ld*(l[i-1]+1)) * v[i-1] / (2*d-g11[1])) > v[i]) if i > 1
            else (w[i-1] * d > v[i]))):
        i -= 1
        # merge two pools
        l[i] += l[i+1]
        if i > 0:
            ll = min(l[i] - 1, 999)  # precomputed kernel shorter than ISI -> simply truncate
            v[i] = (np.ascontiguousarray(g11[:ll + 1]).dot(
                    np.ascontiguousarray(y[t[i]:t[i]+ll+1]))
                        - g11g12[ll] * w[i-1]) / g11g11[ll]
            w[i] = (g11[ll] * v[i] + g12[ll] * w[i-1])
        else:  # update first pool too instead of taking it granted as true
            ll = min(l[i], 1000)
            v[i] = max(0, np.exp(ld * np.arange(ll)).
                            dot(y[:ll]) * (1-d*d) / (1 - exp(ld*2*ll)))
            w[i] = exp(ld*(ll-1)) * v[i]
    i += 1
    return v,w,t,l,i

@njit('f4[:,:], f4[:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], i4[:,:], i4[:,:], i4[:], i4', parallel=True, cache=True)
def par_fit_next_AR2(y,d,g11,g12,g11g11,g11g12, v,w,t,l,i,tt):
    N = len(y)
    for k in prange(N):
        v[k],w[k],t[k],l[k],i[k] = fit_next_AR2(y[k],d[k],g11[k],g12[k],g11g11[k],g11g12[k], v[k],w[k],t[k],l[k],i[k],tt)
