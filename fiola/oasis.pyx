#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from cython.parallel import prange
from libc.math cimport log, exp, fmax, fmin


cdef int fit_next_AR1(float y, float lg, float[:] v, float[:] w, float[:] l, int i) nogil:
    v[i], w[i], l[i] = y, 1, 1
    cdef float f
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
    return i

def par_fit_next_AR1(float[:] y, float[:] lg, float[:,:] v, float[:,:] w, float[:,:] l, int[:] i):
    cdef int k, N
    N = y.shape[0]
    for k in prange(N, nogil=True):
        i[k] = fit_next_AR1(y[k],lg[k], v[k],w[k],l[k],i[k])


cdef int fit_next_AR2(float[:] y, float d, float[:] g11, float[:] g12, float[:] g11g11, float[:] g11g12,
                 float[:] v, float[:] w, int[:] t, int[:] l, int i, int tt) nogil:
    # [first value, last value, start time, length] of pool
    v[i], w[i], t[i], l[i] = y[tt], y[tt], tt, 1
    cdef float ld, tmp
    cdef int j, ll
    ld = log(d)
    while (i > 0 and  # backtrack until violations fixed
            ((((g11[l[i-1]] * v[i-1] + g12[l[i-1]] * w[i-2]) if l[i-1] < 1000
            else exp(ld*(l[i-1]+1)) * v[i-1] / (2*d-g11[1])) > v[i]) if i > 1
            else (w[i-1] * d > v[i]))):
        i -= 1
        # merge two pools
        l[i] += l[i+1]
        if i > 0:
            ll = int(fmin(l[i] - 1, 999))  # precomputed kernel shorter than ISI -> simply truncate
            tmp = 0
            for j in range(ll+1):
                tmp += g11[j] * y[t[i] + j]           
            v[i] = (tmp - g11g12[ll] * w[i-1]) / g11g11[ll]
            w[i] = (g11[ll] * v[i] + g12[ll] * w[i-1])
        else:  # update first pool too instead of taking it granted as true
            ll = int(fmin(l[i], 1000))
            tmp = 0
            for j in range(ll):
                tmp += exp(ld*j) * y[j]
            v[i] = fmax(0, tmp * (1-d*d) / (1 - exp(ld*2*ll)))
            w[i] = exp(ld*(ll-1)) * v[i]
    i += 1
    return i

def par_fit_next_AR2(float[:,:] y, float[:] d, float[:,:] g11, float[:,:] g12, float[:,:] g11g11, float[:,:] g11g12,
                     float[:,:] v, float[:,:] w, int[:,:] t, int[:,:] l, int[:] i, int tt):
    cdef int k, N
    N = y.shape[0]
    for k in prange(N, nogil=True):
        i[k] = fit_next_AR2(y[k],d[k],g11[k],g12[k],g11g11[k],g11g12[k], v[k],w[k],t[k],l[k],i[k],tt)
