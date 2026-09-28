"""Benchmark Numba fancy indexing and ArrayIterator axis-reduction paths.

Run with numba 0.68.0rc1 (the repo's arraymath.py/arrayobj.py are identical
to that release), from a directory that does not shadow the installed package.
LLM-assisted: see Numba AI tools policy for attribution requirements.
"""

import time

import numpy as np
from numba import jit

REPEAT = 7


@jit
def gather1d(a, idx):
    return a[idx]


@jit
def gather1d_i32(a, idx):
    return a[idx]


@jit
def mask1d(a, m):
    return a[m]


@jit
def rowgather(a, k):
    return a[k, :]


@jit
def colgather(a, k):
    return a[:, k]


@jit
def pointgather(a, r, c):
    return a[r, c]


@jit
def mixed_int_ary(a, k):
    return a[3, k]


@jit
def fancy_nd(a, idx):
    return a[idx]


@jit
def ellipsis_idx(a, k):
    return a[..., k]


@jit
def midgather3d(a, k):
    return a[1, k, :]


@jit
def scatter1d(a, idx, v):
    a[idx] = v


@jit
def maskscatter1d(a, m, v):
    a[m] = v


@jit
def rowscatter(a, k, v):
    a[k, :] = v


@jit
def pointscatter(a, r, c, v):
    a[r, c] = v


@jit
def sum_all(a):
    return np.sum(a)


@jit
def sum_ax0(a):
    return np.sum(a, 0)


@jit
def sum_ax1(a):
    return np.sum(a, 1)


@jit
def sum_ax_tuple(a):
    return np.sum(a, (0, 2))


@jit
def cumsum_ax0(a):
    return np.cumsum(a, 0)


@jit
def cumsum_ax1(a):
    return np.cumsum(a, 1)


@jit
def sum_ax0_3d(a):
    return np.sum(a, 0)


@jit
def sum_ax2_3d(a):
    return np.sum(a, 2)


@jit
def sum_flat(a):
    s = 0.0
    for v in a.flat:
        s += v
    return s


@jit
def sum_fancy_then_sum(a, idx):
    return np.sum(a[idx])


def timeit(fn, *args):
    fn(*args)
    best = float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    n = 2_000_000
    n2 = 2_000
    rng = np.random.default_rng(0)

    a1 = rng.random(n)
    idx1 = rng.integers(0, n, size=n)
    idx1_i32 = idx1.astype(np.int32)
    m1 = rng.random(n) < 0.5

    a2 = rng.random((n2, n2))
    idx2 = rng.integers(0, n2, size=n2)
    idx_big = rng.integers(0, n2, size=n)
    rows = rng.integers(0, n2, size=n)
    cols = rng.integers(0, n2, size=n)

    a2f = np.asfortranarray(a2)

    a3 = rng.random((100, 200, 200))
    idx3 = rng.integers(0, 200, size=200)
    idx2d = rng.integers(0, 100, size=(10, 10))

    v = 1.0
    rowvec = rng.random(n2)

    cases = [
        ("getitem 1D int-array gather (int64)", gather1d, lambda: a1[idx1], (a1, idx1)),
        ("getitem 1D int-array gather (int32 idx)", gather1d_i32, lambda: a1[idx1_i32], (a1, idx1_i32)),
        ("getitem 1D boolean mask", mask1d, lambda: a1[m1], (a1, m1)),
        ("getitem 2D row gather a[k,:]", rowgather, lambda: a2[idx2, :], (a2, idx2)),
        ("getitem 2D col gather a[:,k]", colgather, lambda: a2[:, idx2], (a2, idx2)),
        ("getitem 2D point gather a[r,c]", pointgather, lambda: a2[rows, cols], (a2, rows, cols)),
        ("getitem 2D scalar+array a[3,k]", mixed_int_ary, lambda: a2[3, idx_big], (a2, idx_big)),
        ("getitem 2D 2D-index a[10x10] on 3D", fancy_nd, lambda: a3[idx2d], (a3, idx2d)),
        ("getitem 2D ellipsis+array a[...,k]", ellipsis_idx, lambda: a2[..., idx2], (a2, idx2)),
        ("getitem 3D mid a[1,k,:]", midgather3d, lambda: a3[1, idx3, :], (a3, idx3)),
        ("setitem 1D scatter a[idx]=v", scatter1d, lambda: a1.__setitem__(idx1, v), (a1, idx1, v)),
        ("setitem 1D mask scatter a[m]=v", maskscatter1d, lambda: a1.__setitem__(m1, v), (a1, m1, v)),
        ("setitem 2D row scatter a[k,:]=v", rowscatter, lambda: a2.__setitem__((idx2, slice(None)), rowvec), (a2, idx2, rowvec)),
        ("setitem 2D point scatter a[r,c]=v", pointscatter, lambda: a2.__setitem__((rows, cols), v), (a2, rows, cols, v)),
        ("reduce np.sum(a) [ArrayIterator]", sum_all, lambda: np.sum(a2), (a2,)),
        ("reduce np.sum(a,axis=0) [ArrayIterator]", sum_ax0, lambda: np.sum(a2, 0), (a2,)),
        ("reduce np.sum(a,axis=1) [ArrayIterator]", sum_ax1, lambda: np.sum(a2, 1), (a2,)),
        ("reduce np.sum(a,(0,2)) 3D tuple-axis [ArrayIterator]", sum_ax_tuple, lambda: np.sum(a3, (0, 2)), (a3,)),
        ("reduce np.cumsum(a,axis=0) [ArrayIterator]", cumsum_ax0, lambda: np.cumsum(a2, 0), (a2,)),
        ("reduce np.cumsum(a,axis=1) [ArrayIterator]", cumsum_ax1, lambda: np.cumsum(a2, 1), (a2,)),
        ("reduce np.sum(3D,axis=0) [ArrayIterator]", sum_ax0_3d, lambda: np.sum(a3, 0), (a3,)),
        ("reduce np.sum(3D,axis=2) [ArrayIterator]", sum_ax2_3d, lambda: np.sum(a3, 2), (a3,)),
        ("reduce a.flat loop [ArrayIterator]", sum_flat, lambda: float(a2.sum()), (a2,)),
        ("reduce np.sum(a[idx]) fancy->sum", sum_fancy_then_sum, lambda: np.sum(a1[idx1]), (a1, idx1)),
        ("reduce np.sum(F-order,axis=0) [ArrayIterator]", sum_ax0, lambda: np.sum(a2f, 0), (a2f,)),
    ]

    results = []
    for name, nfn, pfn, args in cases:
        try:
            t_nb = timeit(nfn, *args)
            t_np = timeit(pfn)
        except Exception as e:
            print(f"SKIP {name}: {type(e).__name__}: {e}")
            continue
        results.append((name, t_nb, t_np))

    results.sort(key=lambda r: r[1], reverse=True)
    print(f"{'case':58s} {'numba_ms':>10s} {'numpy_ms':>10s} {'ratio':>7s}")
    for name, t_nb, t_np in results:
        print(f"{name:58s} {t_nb:10.3f} {t_np:10.3f} {t_nb / t_np:7.2f}x")


if __name__ == "__main__":
    main()
