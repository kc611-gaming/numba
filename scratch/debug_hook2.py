import re
import llvmlite.binding as llb
llb.load_library_permanently("libmvec.so.1")

import numba.core.codegen as cg

_orig_opt = cg.CPUCodeLibrary._optimize_final_module
_orig_fin = cg.CPUCodeLibrary._finalize_final_module
print("classes:", cg.CPUCodeLibrary, cg.JITCodeLibrary,
      "_optimize_final_module defined on JITCodeLibrary:",
      "_optimize_final_module" in cg.JITCodeLibrary.__dict__)

MAP = {"llvm.exp.v8f64": "_ZGVeN8v_exp", "llvm.exp.v4f64": "_ZGVdN4v_exp",
       "llvm.exp.v2f64": "_ZGVbN2v_exp"}


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
    print("  patch_module: found=", found)
    if not found:
        return mod
    for d in set(decls):
        ir += d + "\n"
    new = llb.parse_assembly(ir)
    new.verify()
    return new


def _optimize_final_module_patched(self):
    print(">> patched _optimize_final_module called for", self.name)
    self._finalize_final_module = _patched_finalize
    _orig_opt(self)


def _patched_finalize(self):
    self._final_module = patch_module(self._final_module)
    _orig_fin(self)


cg.CPUCodeLibrary._optimize_final_module = _optimize_final_module_patched

import numpy as np
from numba import njit


@njit(fastmath=True)
def np_exp_f(x):
    return np.exp(x)


N = 1 << 16
x = np.random.random(N)
np.testing.assert_allclose(np_exp_f(x), np.exp(x), rtol=1e-13)
print("OK")
