from numba import njit, prange
from math import exp

@njit('f4, f4, f4[:], f4[:], f4[:], i4', cache=True)
def fit_next(y,lg, v,w,l,i):
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
def par_fit_next(y,lg, v,w,l,i):
    N = len(y)
    for k in prange(N):
        v[k],w[k],l[k],i[k] = fit_next(y[k],lg[k], v[k],w[k],l[k],i[k])
