import numpy as np
import time
from numba import njit

N = 128
rng = np.random.default_rng(1)
a3 = rng.random((N, N, N))
a3_nc = a3.transpose((2, 0, 1))  # issue #9819 layout


@njit
def sum_builtin(x):
    return np.sum(x)


@njit
def sum_copy(x):
    return np.sum(np.ascontiguousarray(x))


@njit(fastmath=True)
def sum_memory_order(x):
    # iterate in memory order: inner loop over axis with smallest stride
    sh = x.shape
    st = x.strides
    inner = 0
    if abs(st[1]) < abs(st[inner]):
        inner = 1
    if abs(st[2]) < abs(st[inner]):
        inner = 2
    others = [k for k in range(3) if k != inner]
    total = 0.0
    for i0 in range(sh[others[0]]):
        for i1 in range(sh[others[1]]):
            for i2 in range(sh[inner]):
                idx = [0, 0, 0]
                idx[others[0]] = i0
                idx[others[1]] = i1
                idx[inner] = i2
                total += x[idx[0], idx[1], idx[2]]
    return total


@njit(fastmath=True)
def sum_memory_order_mc(x):
    # manual contiguous inner reduction with vectorizable accumulate
    sh = x.shape
    st = x.strides
    inner = 0
    if abs(st[1]) < abs(st[inner]):
        inner = 1
    if abs(st[2]) < abs(st[inner]):
        inner = 2
    others = [k for k in range(3) if k != inner]
    total = 0.0
    for i0 in range(sh[others[0]]):
        for i1 in range(sh[others[1]]):
            acc = 0.0
            for i2 in range(sh[inner]):
                idx = [0, 0, 0]
                idx[others[0]] = i0
                idx[others[1]] = i1
                idx[inner] = i2
                acc += x[idx[0], idx[1], idx[2]]
            total += acc
    return total


def bench(fn, *a, n=10):
    fn(*a)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


ref = np.sum(a3_nc)
assert np.isclose(sum_builtin(a3_nc), ref)
assert np.isclose(sum_copy(a3_nc), ref)
assert np.isclose(sum_memory_order(a3_nc), ref)
assert np.isclose(sum_memory_order_mc(a3_nc), ref)

print(f"numpy sum (transposed)      : {bench(np.sum, a3_nc)*1e3:8.1f} us")
print(f"numba builtin (patched)     : {bench(sum_builtin, a3_nc)*1e3:8.1f} us")
print(f"numba copy+sum              : {bench(sum_copy, a3_nc)*1e3:8.1f} us")
print(f"numba memory-order loop     : {bench(sum_memory_order, a3_nc)*1e3:8.1f} us")
print(f"numba memory-order + localacc: {bench(sum_memory_order_mc, a3_nc)*1e3:8.1f} us")
