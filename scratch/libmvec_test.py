import llvmlite.binding as llvm
llvm.set_option('', '--vector-library=LIBMVEC-X86')

import numpy as np
from numba import njit
import math
import time


@njit(fastmath=True)
def v_exp(x, out):
    for i in range(x.size):
        out[i] = math.exp(x[i])


@njit(fastmath=True)
def v_exp2(x, out):
    for i in range(x.size):
        out[i] = math.exp(x[i])


x = np.random.random(1024)
out = np.empty_like(x)
v_exp(x, out)

sig = list(v_exp.inspect_asm().keys())[0]
asm = v_exp.inspect_asm(sig)
hits = [l.strip() for l in asm.splitlines() if '_ZGV' in l or 'zmm' in l and ('call' in l or 'vexp' in l)]
print("libmvec/vector hits:")
for h in hits[:10]:
    print("  ", h)
print("has _ZGVdN4v_exp:", '_ZGVdN4v_exp' in asm, "| _ZGVeN8v_exp:", '_ZGVeN8v_exp' in asm)

N = 1 << 20
xb = np.random.random(N)
ob = np.empty_like(xb)
v_exp(xb, ob)
t0 = time.perf_counter()
for _ in range(10):
    v_exp(xb, ob)
dt = (time.perf_counter() - t0) / 10
print(f"manual loop w/ libmvec option: {dt*1e6:.1f} us   (numpy np.exp was ~954 us)")

# also the np.exp ufunc path
@njit(fastmath=True)
def np_exp(x):
    return np.exp(x)

np_exp(xb)
asm2 = np_exp.inspect_asm(list(np_exp.inspect_llvm().keys())[0])
print("np.exp uses _ZGV:", '_ZGV' in asm2)
