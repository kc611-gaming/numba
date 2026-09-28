import numpy as np
import numba
from numba import njit

N = 1 << 20
rng = np.random.default_rng(0)

x64 = rng.random(N)
x32 = x64.astype(np.float32)


def bench(fn, *args, n=20):
    fn(*args)
    import time
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


@njit
def j_exp(x):
    return np.exp(x)


@njit(fastmath=True)
def j_exp_fast(x):
    return np.exp(x)


@njit
def j_log(x):
    return np.log(x)


@njit(fastmath=True)
def j_log_fast(x):
    return np.log(x)


@njit
def j_sum(x):
    return np.sum(x)


@njit(fastmath=True)
def j_sum_fast(x):
    return np.sum(x)


@njit
def j_sort(x):
    return np.sort(x)


@njit
def j_cumsum(x):
    return np.cumsum(x)


@njit
def j_squaresum(x):
    return np.sum(x * x)


@njit(fastmath=True)
def j_squaresum_fast(x):
    return np.sum(x * x)


@njit
def j_manual_squaresum(x):
    acc = 0.0
    for i in range(x.size):
        acc += x[i] * x[i]
    return acc


@njit(fastmath=True)
def j_manual_squaresum_fast(x):
    acc = 0.0
    for i in range(x.size):
        acc += x[i] * x[i]
    return acc


Nt = 128
a3 = rng.random((Nt, Nt, Nt))
a3_nc = a3.transpose((2, 0, 1))  # non-contiguous like issue #9819

print("== np.exp float64 N=1M (us) ==")
print("numpy        ", bench(np.exp, x64, n=10) * 1e3)
print("njit         ", bench(j_exp, x64, n=10) * 1e3)
print("njit fastmath", bench(j_exp_fast, x64, n=10) * 1e3)

print("== np.log float64 N=1M (us) ==")
print("numpy        ", bench(np.log, x64, n=10) * 1e3)
print("njit         ", bench(j_log, x64, n=10) * 1e3)
print("njit fastmath", bench(j_log_fast, x64, n=10) * 1e3)

print("== np.sum float64 N=1M (us) ==")
print("numpy        ", bench(np.sum, x64, n=20))
print("njit         ", bench(j_sum, x64, n=20))
print("njit fastmath", bench(j_sum_fast, x64, n=20))

print("== np.sum of x*x float64 N=1M (us) ==")
print("numpy        ", bench(lambda v: np.sum(v * v), x64, n=10))
print("njit overload", bench(j_squaresum, x64, n=10))
print("njit fastmath", bench(j_squaresum_fast, x64, n=10))
print("manual loop  ", bench(j_manual_squaresum, x64, n=10))
print("manual fast  ", bench(j_manual_squaresum_fast, x64, n=10))

print("== np.sum 3d transposed (non-contiguous) N=128^3 (us) #9819 ==")
print("numpy        ", bench(np.sum, a3_nc, n=10))
print("njit         ", bench(j_sum, a3_nc, n=10))
print("njit fastmath", bench(j_sum_fast, a3_nc, n=10))
print("== np.sum 3d C-contiguous ==")
print("numpy        ", bench(np.sum, a3, n=10))
print("njit         ", bench(j_sum, a3, n=10))

print("== np.sort float64 N=1M (us) #5373/#9611 ==")
print("numpy        ", bench(np.sort, x64, n=5))
print("njit         ", bench(j_sort, x64, n=5))

print("== np.cumsum float64 N=1M (us) ==")
print("numpy        ", bench(np.cumsum, x64, n=10))
print("njit         ", bench(j_cumsum, x64, n=10))
