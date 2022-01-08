from multiprocessing.pool import Pool, ThreadPool
from math import exp
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
for i in range(2, T):
    C[:, i] += g * C[:, i - 1]
Y = C + sn * np.random.randn(N, T).astype(np.float32)

reps = 2
print('Process 1 frame at a time\nNumba')
for _ in range(reps):
    lg = np.log(g)
    v, w, l, s = np.zeros((4, N, T), dtype=np.float32)#'f4')
    i = np.zeros(N, dtype=np.int32)  # index of last pool
    t = -time() 
    for y in Y.T:
        par_fit_next(y,lg*np.ones(N, dtype=np.float32), v,w,l,i)
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
