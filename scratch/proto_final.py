"""Final prototype: libmvec vector math via IR rewrite of vectorized
math intrinsics (post-optimization, pre-codegen), on top of patches A+B.
"""
import llvmlite.binding as llb
llb.load_library_permanently("libmvec.so.1")

import numba.core.codegen as cg

MAP = {
    # double, SSE2 / AVX2 / AVX512 (glibc libmvec ABI)
    "llvm.exp.v2f64": "_ZGVbN2v_exp", "llvm.exp.v4f64": "_ZGVdN4v_exp",
    "llvm.exp.v8f64": "_ZGVeN8v_exp",
    "llvm.log.v2f64": "_ZGVbN2v_log", "llvm.log.v4f64": "_ZGVdN4v_log",
    "llvm.log.v8f64": "_ZGVeN8v_log",
    "llvm.sin.v2f64": "_ZGVbN2v_sin", "llvm.sin.v4f64": "_ZGVdN4v_sin",
    "llvm.sin.v8f64": "_ZGVeN8v_sin",
    "llvm.cos.v2f64": "_ZGVbN2v_cos", "llvm.cos.v4f64": "_ZGVdN4v_cos",
    "llvm.cos.v8f64": "_ZGVeN8v_cos",
    "llvm.exp.v4f32": "_ZGVbN4v_expf", "llvm.exp.v8f32": "_ZGVdN8v_expf",
    "llvm.exp.v16f32": "_ZGVeN16v_expf",
    "llvm.log.v4f32": "_ZGVbN4v_logf", "llvm.log.v8f32": "_ZGVdN8v_logf",
    "llvm.log.v16f32": "_ZGVeN16v_logf",
    "llvm.sin.v4f32": "_ZGVbN4v_sinf", "llvm.sin.v8f32": "_ZGVdN8v_sinf",
    "llvm.sin.v16f32": "_ZGVeN16v_sinf",
    "llvm.cos.v4f32": "_ZGVbN4v_cosf", "llvm.cos.v8f32": "_ZGVdN8v_cosf",
    "llvm.cos.v16f32": "_ZGVeN16v_cosf",
}


def patch_module(mod):
    ir = str(mod)
    changed = False
    for intrin, vecfn in MAP.items():
        if "@" + intrin in ir:
            # renaming also renames the `declare` line, which then serves
            # as the declaration for the libmvec symbol
            ir = ir.replace("@" + intrin, "@" + vecfn)
            changed = True
    if not changed:
        return mod
    new = llb.parse_assembly(ir)
    new.verify()
    return new


_orig_fin = cg.CPUCodeLibrary._finalize_final_module


def _finalize_final_module_patched(self):
    self._final_module = patch_module(self._final_module)
    _orig_fin(self)


cg.CPUCodeLibrary._finalize_final_module = _finalize_final_module_patched

# ---------------- benchmark ----------------
import numpy as np
import time
from numba import njit


@njit(fastmath=True)
def np_exp_f(x):
    return np.exp(x)


@njit(fastmath=True)
def np_log_f(x):
    return np.log(x)


@njit(fastmath=True)
def np_exp_f32(x):
    return np.exp(x)


@njit
def np_sum_s(x):
    return np.sum(x)


@njit(fastmath=True)
def np_sum_f(x):
    return np.sum(x)


N = 1 << 20
xb = np.random.random(N)
x32 = xb.astype(np.float32)

np.testing.assert_allclose(np_exp_f(xb), np.exp(xb), rtol=1e-13)
np.testing.assert_allclose(np_log_f(xb), np.log(xb), rtol=1e-12)
np.testing.assert_allclose(np_exp_f32(x32), np.exp(x32), rtol=1e-6)
np.testing.assert_allclose(np_sum_f(xb), xb.sum(), rtol=1e-12)
print("correctness OK")


def bench(fn, *a, n=10):
    fn(*a)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e6


for name, fn, ref in [
    ("np.exp f64 fastmath", np_exp_f, np.exp),
    ("np.log f64 fastmath", np_log_f, np.log),
    ("np.exp f32 fastmath", np_exp_f32, np.exp),
    ("np.sum f64 strict  ", np_sum_s, None),
    ("np.sum f64 fastmath", np_sum_f, None),
]:
    dt = bench(fn, xb if "f32" not in name else x32)
    if ref is not None:
        dtn = bench(ref, xb if "f32" not in name else x32)
        print(f"{name}: {dt*1e3:8.1f} us | numpy {dtn*1e3:8.1f} us | ratio {dt/dtn:.2f}")
    else:
        print(f"{name}: {dt*1e3:8.1f} us")

sig = list(np_exp_f.inspect_llvm().keys())[0]
asm = np_exp_f.inspect_asm(sig)
print("np.exp f64 asm has _ZGV:", "_ZGV" in asm)
