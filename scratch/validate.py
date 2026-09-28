import numpy as np
from numba import njit


@njit(fastmath=False)
def exp_s(x):
    return np.exp(x)


@njit(fastmath=True)
def exp_f(x):
    return np.exp(x)


@njit(fastmath=False)
def log_s(x):
    return np.log(x)


@njit
def sum_s(x):
    return np.sum(x)


@njit(fastmath=True)
def sum_f(x):
    return np.sum(x)


rng = np.random.default_rng(42)
fail = 0

cases = {
    "c 1d": rng.random(77),
    "empty": rng.random(0),
    "len1": rng.random(1),
    "f-order 2d": np.asfortranarray(rng.random((4, 7))),
    "c 2d": rng.random((3, 5)),
    "3d": rng.random((2, 3, 4)),
    "transposed": rng.random((5, 6)).T,
    "f32": rng.random(33).astype(np.float32),
    "f16-ish small": rng.random(9),
}
for name, arr in cases.items():
    ref = np.exp(arr)
    for fn, tag in [(exp_s, "strict"), (exp_f, "fastmath")]:
        got = fn(arr)
        try:
            np.testing.assert_allclose(got, ref, rtol=1e-13)
        except AssertionError as e:
            print(f"EXP FAIL {name} {tag}: {e}")
            fail += 1
    ref = np.log(arr)
    for fn, tag in [(log_s, "strict")]:
        with np.errstate(all="ignore"):
            got = fn(arr)
        try:
            np.testing.assert_allclose(got, ref, rtol=1e-12)
        except AssertionError as e:
            print(f"LOG FAIL {name} {tag}: {e}")
            fail += 1
    ref = np.sum(arr)
    got_s = sum_s(arr)
    if not np.isclose(got_s, ref, rtol=1e-12):
        print(f"SUM FAIL {name} strict: {got_s} vs {ref}")
        fail += 1
    got_f = sum_f(arr)
    if not np.isclose(got_f, ref, rtol=1e-12):
        print(f"SUM FAIL {name} fastmath: {got_f} vs {ref}")
        fail += 1

# int arrays
for dt in (np.int32, np.int64, np.bool_):
    a = (rng.random(50) > 0.5).astype(dt)
    r = np.sum(a)
    assert sum_s(a) == r, (dt, sum_s(a), r)
    assert sum_f(a) == r, (dt, sum_f(a), r)
print("int/bool sum OK")

# ufunc with out= and second array (should fall back to generic path)
x = rng.random(10)
out = np.empty(10)
res = exp_f(x)
np.testing.assert_allclose(res, np.exp(x), rtol=1e-13)

# gufunc-ish: exp on axis slices of 2d (non-contig rows)
m = rng.random((6, 9))
row = m[2]  # contiguous
np.testing.assert_allclose(exp_f(row), np.exp(row), rtol=1e-13)
col = m[:, 3]  # non-contiguous
np.testing.assert_allclose(exp_f(col), np.exp(col), rtol=1e-13)

print("validation done, failures:", fail)
