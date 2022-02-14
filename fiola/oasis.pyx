#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

from cython.parallel import prange
from libc.math cimport log, exp, fmax, fmin


cdef int fit_next_AR1(float y, float lg, float[:] v, float[:] w,
                      int[:] t, int[:] l, float[:] h, int i, int tt) nogil:
    v[i], w[i], t[i], l[i] = y, 1, tt, 1
    cdef float f
    if i > 0:
        f = exp(lg * l[i-1])
        h[i] = y if (i==1 and v[0]<0) else y - v[i-1] / w[i-1] * f
    while (i > 0 and h[i] < 0): # backtrack until violations fixed
        i -= 1
        # merge two pools
        v[i] += v[i+1] * f
        w[i] += w[i+1] * f*f
        l[i] += l[i+1]
        if i > 0:
            f = exp(lg * l[i-1])
            h[i] = v[i] / w[i] if (i==1 and v[0]<0) else v[i] / w[i] - v[i-1] / w[i-1] * f
    i += 1
    return i

def par_fit_next_AR1(float[:] y, float[:] lg, float[:,:] v, float[:,:] w,
                     int[:,:] t, int[:,:] l, float[:,:] h, int[:] i, int tt):
    cdef int k, N
    N = y.shape[0]
    for k in prange(N, nogil=True):
        i[k] = fit_next_AR1(y[k],lg[k], v[k],w[k],t[k],l[k],h[k],i[k],tt)


cdef int fit_next_AR2(float[:] y, float d, float[:] g11, float[:] g12, float[:] g11g11, float[:] g11g12,
                      float[:] v, float[:] w, int[:] t, int[:] l, float[:] h, int i, int tt) nogil:
    # [first value, last value, start time, length] of pool
    v[i], w[i], t[i], l[i] = y[tt], y[tt], tt, 1
    cdef float ld, tmp
    cdef int j, ll
    ld = log(d)
    if i > 0:
        h[i] = v[i] - d*w[i-1] if i==1 \
            else v[i] - ((g11[l[i-1]] * v[i-1] + g12[l[i-1]] * w[i-2]) if l[i-1] < 1000
                        else exp(ld*(l[i-1]+1)) * v[i-1] / (2*d-g11[1]))
    while (i > 0 and h[i] < 0):  # backtrack until violations fixed
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
            h[i] = v[i] - d*w[i-1] if i==1 \
                else v[i] - ((g11[l[i-1]] * v[i-1] + g12[l[i-1]] * w[i-2]) if l[i-1] < 1000
                            else exp(ld*(l[i-1]+1)) * v[i-1] / (2*d-g11[1]))
        else:  # update first pool
            ll = int(fmin(l[i], 1000))
            tmp = 0
            for j in range(ll):
                tmp += exp(ld*j) * y[j]
            v[i] = fmax(0, tmp * (1-d*d) / (1 - exp(ld*2*ll)))
            w[i] = exp(ld*(ll-1)) * v[i]
    i += 1
    return i

def par_fit_next_AR2(float[:,:] y, float[:] d, float[:,:] g11, float[:,:] g12, float[:,:] g11g11, float[:,:] g11g12,
                     float[:,:] v, float[:,:] w, int[:,:] t, int[:,:] l, float[:,:] h, int[:] i, int tt):
    cdef int k, N
    N = y.shape[0]
    for k in prange(N, nogil=True):
        i[k] = fit_next_AR2(y[k],d[k],g11[k],g12[k],g11g11[k],g11g12[k], v[k],w[k],t[k],l[k],h[k],i[k],tt)



def reconstruct_AR1(float[:,:] s, float[:,:] c, float[:] g,
                     float[:,:] v, float[:,:] w, int[:,:] t, float[:,:] h, int[:] i):
    cdef int n, k, N, T
    N = s.shape[0]
    T = s.shape[1]
    for n in prange(N, nogil=True):
        c[n,0] = fmax(v[n,0]/w[n,0], 0)
        for k in range(1,i[n]):
            s[n,t[n,k]] = h[n,k]
            c[n,t[n,k]] = h[n,k]
        for k in range(T-1):
            c[n,k+1] += g[n]*c[n,k]

def reconstruct_AR2(float[:,:] s, float[:,:] c, float[:] d, float[:,:] g,
                     float[:,:] v, float[:,:] w, int[:,:] t, int[:,:] l, float[:,:] h, int[:] i):
    cdef int n, k, j, N
    N = v.shape[0]
    for n in prange(N, nogil=True):
        for k in range(i[n]):
            s[n,t[n,k]] = h[n,k]
        c[n,0] = fmax(v[n,0], 0)
        for j in range(1,l[n,0]):
            c[n,j] = d[n]*c[n,j-1]
        for k in range(1, i[n]):
            c[n,t[n,k]] = v[n,k]
            for j in range(t[n,k]+1, t[n,k]+l[n,k]-1):
                c[n,j] = g[n,0] * c[n,j-1] + g[n,1] * c[n,j-2]
            c[n,t[n,k]+l[n,k]-1] = w[n,k]

