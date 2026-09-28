import re
import llvmlite.binding as llb
llb.load_library_permanently("libmvec.so.1")

import numba.core.codegen as cg
import numpy as np
from numba import njit

patched = []

_orig_opt = cg.CPUCodeLibrary._optimize_final_module
_orig_fin = cg.CPUCodeLibrary._finalize_final_module


def patch_module(mod):
    ir = str(mod)
    hits = []
    m = re.search(r"declare <\d+ x [a-z0-9]+> @llvm\.\w+\.v\d+f\d+", ir)
    hits.append(m.group(0) if m else None)
    found = False
    decls = []
    for intrin, vecfn in [("llvm.exp.v8f64", "_ZGVeN8v_exp"), ("llvm.exp.v4f64", "_ZGVdN4v_exp"),
                          ("llvm.exp.v2f64", "_ZGVbN2v_exp")]:
        if "@" + intrin in ir:
            found = True
            mm = re.search(r"declare (<\d+ x [a-z0-9]+>) @" + re.escape(intrin), ir)
            if mm:
                decls.append(f"declare {mm.group(1)} @{vecfn}({mm.group(1)})")
            ir = ir.replace("@" + intrin, "@" + vecfn)
    print("  patch_module: found=", found, "intrinsics_seen=", hits)
    if not found:
        return mod
    for d in set(decls):
        ir += d + "\n"
    new = llb.parse_assembly(ir)
    new.verify()
    return new


def _optimize_final_module_patched(self):
    def finalize_wrapper():
        try:
            self._final_module = patch_module(self._final_module)
        except Exception as e:
            print("patch_module failed:", e)
        _orig_fin(self)
    self._finalize_final_module = finalize_wrapper
    try:
        _orig_opt(self)
    finally:
        del self._finalize_final_module


cg.CPUCodeLibrary._optimize_final_module = _optimize_final_module_patched


@njit(fastmath=True)
def np_exp_f(x):
    return np.exp(x)


N = 1 << 16
x = np.random.random(N)
r = np_exp_f(x)
np.testing.assert_allclose(r, np.exp(x), rtol=1e-13)
print("OK")

sig = list(np_exp_f.inspect_llvm().keys())[0]
ir = np_exp_f.inspect_llvm(sig)
for l in ir.splitlines():
    if 'exp' in l and 'call' in l:
        print(l[:140])
asm = np_exp_f.inspect_asm(sig)
print("_ZGV in asm:", '_ZGV' in asm)
