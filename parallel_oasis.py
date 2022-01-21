from multiprocessing.pool import Pool, ThreadPool
from math import exp, log, sqrt
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np
from time import time
from caiman.source_extraction.cnmf.oasis import OASIS

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

# generate data
N, T = 400, 1000
g, sn = .95, .1
lg = np.log(g)
firerate, framerate = .2, 30
np.random.seed(0)
S = np.random.rand(N, T) < firerate / framerate
C = S.astype(np.float32)
for i in range(1, T):
    C[:, i] += g * C[:, i - 1]
Y = C + sn * np.random.randn(N, T).astype(np.float32)

reps = 2
print('AR1\n1 neuron\nNumba')
for _ in range(reps):
    lg = np.log(g)
    v, w, l = np.zeros((3, 50), dtype=np.float32)
    i = 0
    t = -time() 
    for y in Y[0]:
        fit_next(y,lg, v,w,l,i)
        tmp = len(v)
        if i>=tmp:
            vwl = np.zeros((3, tmp+50), dtype=np.float32)
            vwl[:,:,:tmp] = v,w,l
            v, w, l = vwl
    t += time()
    print('Time per frame %.4fμs' % (t/T*1e6))

print('Cython')
for _ in range(reps):
    o = OASIS(g=g)
    t = -time()
    for y in Y[0]:
        o.fit_next(y)
    t += time()
    print('Time per frame %.4fμs' % (t/T*1e6))

print('\n%g neurons\nNumba' % N)
for _ in range(reps):
    lg = np.log(g)
    v, w, l = np.zeros((3, N, 50), dtype=np.float32)
    i = np.zeros(N, dtype=np.int32)  # index of last pool
    t = -time() 
    for y in Y.T:
        par_fit_next(y,lg*np.ones(N, dtype=np.float32), v,w,l,i)
        tmp = v.shape[1]
        if i.max()>=tmp:
            vwl = np.zeros((3, N, tmp+50), dtype=np.float32)
            vwl[:,:,:tmp] = v,w,l
            v, w, l = vwl
    t += time()
    print('Time per frame %.4fμs' % (t/T*1e6))
    print('Time per frame per neuron %.4fμs' % (t/T/N*1e6))

print('Cython')
for _ in range(reps):
    OASISinstances = [OASIS(g=g) for _ in range(N)]
    t = -time()
    for y in Y.T:
        for o, y_ in zip(OASISinstances, y):
            o.fit_next(y_)
    t += time()
    print('Time per frame %.4fμs' % (t/T*1e6))
    print('Time per frame per neuron %.4fμs' % (t/T/N*1e6))


# construct s and c
s = np.zeros((N, T), dtype=np.float32)
v /= w
v[v < 0] = 0
for k,ii in enumerate(i):
    t = np.cumsum(l[k,:ii-1]).astype(np.uint32)
    s[k,t] = v[k,1:ii] - v[k,:ii-1] * np.exp(lg * l[k,:ii-1])
c = s.copy()
for j in range(T-1):
    c[:,j+1] += g*c[:,j]

plt.plot(Y[0])
plt.plot(c[0])
plt.plot(OASISinstances[0].c, ':')
plt.show()



@njit('f4[:], f4, f4[:], f4[:], f4[:], f4[:], f4[:], f4[:], i4[:], i4[:], i4, i4', cache=True)
def fit_next_AR2(y,d,g11,g12,g11g11,g11g12, v,w,t,l,i,tt):
    # [first value, last value, start time, length] of pool
    v[i], w[i], t[i], l[i] = y[tt], y[tt], tt, 1 
    while (i > 0 and  # backtrack until violations fixed
            ((g11[l[i-1]] * v[i-1] + g12[l[i-1]] * w[i-2] > v[i]) if i > 1
            else (w[i-1] * d > v[i]))):
        i -= 1
        # merge two pools
        l[i] += l[i+1]
        ll = l[i] - 1
        if i > 0:
            v[i] = (g11[:ll + 1].dot(y[t[i]:t[i] + l[i]])
                        - g11g12[ll] * w[i-1]) / g11g11[ll]
            w[i] = (g11[ll] * v[i] + g12[ll] * w[i-1])
        else:  # update first pool too instead of taking it granted as true
            ld = log(d)
            v[i] = max(0, np.exp(ld * np.arange(ll+1)).
                            dot(y[:l[i]]) * (1-d*d) / (1 - exp(ld*2*(ll+1))))
            w[i] = exp(ld*ll) * v[i]
    i += 1
    return v,w,t,l,i

@njit('f4[:,:], f4[:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], f4[:,:], i4[:,:], i4[:,:], i4[:], i4', parallel=True, cache=True)
def par_fit_next_AR2(y,d,g11,g12,g11g11,g11g12, v,w,t,l,i,tt):
    N = len(y)
    for k in prange(N):
        v[k],w[k],t[k],l[k],i[k] = fit_next_AR2(y[k],d[k],g11[k],g12[k],g11g11[k],g11g12[k], v[k],w[k],t[k],l[k],i[k],tt)

# generate data
N, T = 400, 1000
g1, g2, sn = 1.85, -.855, .3 # d,r = .95,.9
firerate, framerate = .2, 30
np.random.seed(0)
S = np.random.rand(N, T) < firerate / framerate
C = S.astype(np.float32)
for i in range(2, T):
    C[:, i] += g1 * C[:, i-1] + g2 * C[:, i-2]
Y = C + sn * np.random.randn(N, T).astype(np.float32)

# precompute
d = (g1 + sqrt(g1 * g1 + 4 * g2)) / 2
r = (g1 - sqrt(g1 * g1 + 4 * g2)) / 2
g11 = ((np.exp(log(d) * np.arange(1, 1001)) -
        np.exp(log(r) * np.arange(1, 1001))) / (d - r)).astype(np.float32)
g12 = np.zeros(1000, dtype=np.float32)
g12[1:] = g2 * g11[:-1]
g11g11 = np.cumsum(g11 * g11)
g11g12 = np.cumsum(g11 * g12)

reps = 2
print('\n\nAR2\n1 neuron\nNumba')
for _ in range(reps):
    tt = 0
    y = np.zeros(1000, dtype=np.float32)
    v, w = np.zeros((2, T), dtype=np.float32)
    t, l = np.zeros((2, T), dtype=np.int32)
    i = 0 # np.zeros(N, dtype=np.int32)  # index of last pool
    t_ = -time() 
    for yt in Y[0]:
        y[tt] = yt 
        v,w,t,l,i = fit_next_AR2(y,d,g11,g12,g11g11,g11g12, v,w,t,l,i,tt)
        tt +=1
    t_ += time()
    print('Time per frame %.4fμs' % (t_/T*1e6))

print('Cython')
for _ in range(reps):
    o = OASIS(g=g1,g2=g2)
    t = -time()
    for y_ in Y[0]:
        o.fit_next(y_)
    t += time()
    print('Time per frame %.4fμs' % (t/T*1e6))

print('\n%g neurons\nNumba' % N)
d_ = d*np.ones(N, dtype=np.float32)
g11_ = np.outer(np.ones(N, dtype=np.float32), g11)
g12_ = np.outer(np.ones(N, dtype=np.float32), g12)
g11g11_ = np.outer(np.ones(N, dtype=np.float32), g11g11)
g11g12_ = np.outer(np.ones(N, dtype=np.float32), g11g12)
for _ in range(reps):
    tt = 0
    y = np.zeros((N, T), dtype=np.float32)
    v, w = np.zeros((2, N, T), dtype=np.float32)
    t, l = np.zeros((2, N, T), dtype=np.int32)
    i = np.zeros(N, dtype=np.int32)  # index of last pool
    t_ = -time() 
    for yt in Y.T:
        y[:,tt] = yt 
        par_fit_next_AR2(y,d_, g11_,g12_,g11g11_,g11g12_, v,w,t,l,i,tt)
        tt +=1
    t_ += time()
    print('Time per frame %.4fμs' % (t_/T*1e6))
    print('Time per frame per neuron %.4fμs' % (t_/T/N*1e6))

print('Cython')
for _ in range(reps):
    OASISinstances = [OASIS(g=g1,g2=g2) for _ in range(N)]
    t_ = -time()
    for y in Y.T:
        for o, y_ in zip(OASISinstances, y):
            o.fit_next(y_)
    t_ += time()
    print('Time per frame %.4fμs' % (t_/T*1e6))
    print('Time per frame per neuron %.4fμs' % (t_/T/N*1e6))


# construct c
c = np.empty((N, T), dtype=np.float32)
for n in range(N):
    tmp = max(v[n,0], 0)
    for j in range(l[n,0]):
        c[n,j] = tmp
        tmp *= d_[n]
for n in range(N):
    for k in range(1, i[n]+1):
        c[n,t[n,k]] = v[n,k]
        for j in range(t[n,k]+1, t[n,k]+l[n,k]-1):
            c[n,j] = g1 * c[n,j-1] + g2 * c[n,j-2]
        c[n,t[n,k]+l[n,k]-1] = w[n,k]
# construct s
s = c.copy()
s[:,:2] = 0
s[:,2:] -= (g1 * c[:,1:-1] + g2 * c[:,:-2])

plt.plot(Y[0])
plt.plot(c[0])
plt.plot(OASISinstances[0].c, ':')
plt.show()
