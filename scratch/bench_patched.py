"""Benchmark the fully-patched numba: flat ufunc loop + index-loop sum +
libmvec vector math rewrite."""
import ctypes
import llvmlite.binding as llb

try:
    llb.load_library_permanently("libmvec.so.1")
    HAVE_LIBMVEC = True
except Exception as e:
    HAVE_LIBMVEC = False
    print("no libmvec:", e)

MAP = {
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
    found = False
    decls = []
    for intrin, vecfn in MAP.items():
        if "@" + intrin in ir:
            found = True
            m = re.search(r"declare (<\d+ x [a-z0-9]+>) @" + re.escape(intrin), ir)
            if m:
                decls.append(f"declare {m.group(1)} @{vecfn}({m.group(1)})")
            ir = ir.replace("@" + intrin, "@" + vecfn)
    if not found:
        return mod
    for d in set(decls):
        ir += d + "\n"
    new = llb.parse_assembly(ir)
    new.verify()
    return new


import numba.core.codegen as cg

_orig_opt = cg.CPUCodeLibrary._optimize_final_module
_orig_fin = cg.CPUCodeLibrary._finalize_final_module


def _optimize_final_module_patched(self):
    def finalize_wrapper():
        self._final_module = patch_module(self._final_module)
        _orig_fin(self)
    self._finalize_final_module = finalize_wrapper
    try:
        _orig_opt(self)
    finally:
        del self._finalize_final_module


cg.CPUCodeLibrary._optimize_final_module = _optimize_final_module_patched

# ---------------- benchmarks ----------------
import numpy as np
import time
from numba import njit


@njit(fastmath=True)
def np_exp_f(x):
    return np.exp(x)


@njit(fastmath=True)
def np_log_f(x):
    return np.log(x)


@njit
def np_exp_s(x):
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


def bench(fn, *a, n=10):
    fn(*a)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e6


ok = np.testing.assert_allclose(np_exp_f(xb), np.exp(xb), rtol=1e-13)
print("np.exp fastmath matches numpy:", True)
np.testing.assert_allclose(np_log_f(xb), np.log(xb), rtol=1e-12)
np.testing.assert_allclose(np_sum_f(xb), xb.sum(), rtol=1e-12)

for name, fn, ref in [
    ("np.exp fastmath", np_exp_f, np.exp),
    ("np.log fastmath", np_log_f, np.log),
    ("np.exp strict  ", np_exp_s, np.exp),
    ("np.sum strict  ", np_sum_s, None),
    ("np.sum fastmath", np_sum_f, None),
]:
    dt = bench(fn, xb)
    if ref is not None:
        dtn = bench(ref, xb)
        print(f"{name}: {dt*1e3:8.1f} us | numpy {dtn*1e3:8.1f} us | ratio {dt/dtn:.2f}")
    else:
        print(f"{name}: {dt*1e3:8.1f} us")

# float32 exp
@njit(fastmath=True)
def np_expf_f(x):
    return np.exp(x)

r = np_exp_f(x32)
np.testing.assert_allclose(r, np.exp(x32), rtol=1e-6)
dt = bench(np_exp_f, x32)
dtn = bench(np.exp, x32)
print(f"np.exp f32 fastmath: {dt*1e3:8.1f} us | numpy {dtn*1e3:8.1f} us | ratio {dt/dtn:.2f}")

sig = list(np_exp_f.inspect_llvm().keys())[0]
asm = np_exp_f.inspect_asm(sig)
print("np.exp fastmath asm _ZGV:", "_ZGV" in asm, "| zmm:", "zmm" in asm)
