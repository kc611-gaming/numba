import numpy as np
from numba import njit
import math


@njit(fastmath=True)
def v_exp(x, out):
    for i in range(x.size):
        out[i] = math.exp(x[i])


x = np.random.random(8)
out = np.empty_like(x)
v_exp(x, out)

sig = list(v_exp.inspect_llvm().keys())[0]
ir = v_exp.inspect_llvm(sig)
for l in ir.splitlines():
    if 'declare' in l or 'exp' in l.lower() and ('call' in l or 'declare' in l):
        print(l[:150])
