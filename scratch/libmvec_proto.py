"""Prototype: libmvec vector math for numba via post-optimization IR rewrite.

Replaces calls to llvm.exp/log/sin/cos vector intrinsics (which the x86
backend scalarizes into per-lane libm calls) with calls into glibc's
libmvec vector math library, in the spirit of numba's old SVML support.
"""
import ctypes
import re

import llvmlite.binding as llb

# Load libmvec so its symbols are resolvable by the JIT.
try:
    llb.load_library_permanently("libmvec.so.1")
    HAVE_LIBMVEC = True
except Exception as e:
    HAVE_LIBMVEC = False
    print("no libmvec:", e)

if HAVE_LIBMVEC:
    for name in ("_ZGVdN4v_exp", "_ZGVeN8v_exp", "_ZGVdN4v_log", "_ZGVeN8v_log"):
        try:
            ctypes.CDLL("libmvec.so.1").__getattr__  # noqa
        except Exception:
            pass
    libmvec = ctypes.CDLL("libmvec.so.1")
    for name in ("_ZGVdN4v_exp", "_ZGVeN8v_exp", "_ZGVdN4v_log", "_ZGVeN8v_log",
                 "_ZGVdN4v_sin", "_ZGVeN8v_sin", "_ZGVdN4v_cos", "_ZGVeN8v_cos"):
        assert hasattr(libmvec, name), name
    print("libmvec symbols present")

MAP_D = {  # double: intrinsic -> libmvec base name
    "llvm.exp.v2f64": "_ZGVbN2v_exp",
    "llvm.exp.v4f64": "_ZGVdN4v_exp",
    "llvm.exp.v8f64": "_ZGVeN8v_exp",
    "llvm.log.v2f64": "_ZGVbN2v_log",
    "llvm.log.v4f64": "_ZGVdN4v_log",
    "llvm.log.v8f64": "_ZGVeN8v_log",
    "llvm.sin.v2f64": "_ZGVbN2v_sin",
    "llvm.sin.v4f64": "_ZGVdN4v_sin",
    "llvm.sin.v8f64": "_ZGVeN8v_sin",
    "llvm.cos.v2f64": "_ZGVbN2v_cos",
    "llvm.cos.v4f64": "_ZGVdN4v_cos",
    "llvm.cos.v8f64": "_ZGVeN8v_cos",
    "llvm.exp.v4f32": "_ZGVbN4v_expf",
    "llvm.exp.v8f32": "_ZGVdN8v_expf",
    "llvm.exp.v16f32": "_ZGVeN16v_expf",
    "llvm.log.v4f32": "_ZGVbN4v_logf",
    "llvm.log.v8f32": "_ZGVdN8v_logf",
    "llvm.log.v16f32": "_ZGVeN16v_logf",
    "llvm.sin.v4f32": "_ZGVbN4v_sinf",
    "llvm.sin.v8f32": "_ZGVdN8v_sinf",
    "llvm.sin.v16f32": "_ZGVeN16v_sinf",
    "llvm.cos.v4f32": "_ZGVbN4v_cosf",
    "llvm.cos.v8f32": "_ZGVdN8v_cosf",
    "llvm.cos.v16f32": "_ZGVeN16v_cosf",
}


def patch_module(mod):
    ir = str(mod)
    found = False
    decls = []
    for intrin, vecfn in MAP_D.items():
        if "@" + intrin in ir:
            found = True
            vt = re.search(r"@%s\(" % intrin, ir)
            # vector type from signature: match "declare <N x ty> @intrin("
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


def _optimize_final_module_patched(self):
    orig_finalize = self._finalize_final_module

    def finalize_wrapper():
        # patch vector math intrinsics right before object emission
        self._final_module = patch_module(self._final_module)
        orig_finalize()
    self._finalize_final_module = finalize_wrapper
    try:
        _orig_opt(self)
    finally:
        self._finalize_final_module = orig_finalize


cg.CPUCodeLibrary._optimize_final_module = _optimize_final_module_patched

# ---------------- benchmarks ----------------
import numpy as np
import time
from numba import njit


@njit(fastmath=True)
def np_exp(x):
    return np.exp(x)


@njit(fastmath=True)
def np_log(x):
    return np.log(x)


@njit(fastmath=True)
def manual_exp(x, out):
    for i in range(x.size):
        out[i] = math_exp(x[i])
    return out


from numba.extending import intrinsic  # noqa: E402
import numba.types as nt


@njit(fastmath=True)
def math_loop_exp(x, out):
    for i in range(x.size):
        out[i] = _exp(x[i])
    return out


from numba.cpython import mathimpl


@njit(fastmath=True)
def _exp(x):
    return mathimpl.exp(x)


N = 1 << 20
xb = np.random.random(N)
ob = np.empty_like(xb)

np.testing.assert_allclose(np_exp(xb), np.exp(xb), rtol=1e-12)
np.testing.assert_allclose(np_log(xb), np.log(xb), rtol=1e-12)

for name, fn, ref in [("np.exp", np_exp, np.exp), ("np.log", np_log, np.log)]:
    fn(xb)
    t0 = time.perf_counter()
    for _ in range(10):
        fn(xb)
    dt = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        ref(xb)
    dt_np = (time.perf_counter() - t0) / 10
    print(f"{name}: numba+libmvec {dt*1e6:8.1f} us | numpy {dt_np*1e6:8.1f} us | ratio {dt/dt_np:.2f}")

sig = list(np_exp.inspect_llvm().keys())[0]
asm = np_exp.inspect_asm(sig)
print("np.exp asm uses _ZGV:", "_ZGV" in asm)
