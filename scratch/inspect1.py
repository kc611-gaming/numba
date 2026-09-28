import numpy as np
from numba import njit

@njit(fastmath=True)
def j_sum_fast(x):
    return np.sum(x)

@njit(fastmath=True)
def j_exp_fast(x):
    return np.exp(x)

x = np.random.random(1024)
j_sum_fast(x)
j_exp_fast(x)

asm_sum = j_sum_fast.inspect_asm(list(j_sum_fast.inspect_asm().keys())[0])
asm_exp = j_exp_fast.inspect_asm(list(j_exp_fast.inspect_asm().keys())[0])

for name, asm in [("sum", asm_sum), ("exp", asm_exp)]:
    vec = [l.strip() for l in asm.splitlines() if ("vaddp" in l or "vmulp" in l or "ymm" in l or "zmm" in l or "call" in l and "exp" in l)]
    print(f"--- {name}: vector/call lines (first 15) ---")
    for l in vec[:15]:
        print("  ", l)
    calls = [l.strip() for l in asm.splitlines() if "call" in l]
    print(f"--- {name}: all calls ---")
    for l in calls[:10]:
        print("  ", l)

sig = list(j_sum_fast.inspect_llvm().keys())[0]
ir = j_sum_fast.inspect_llvm(sig)
print("--- sum IR loop markers ---")
import re
for l in ir.splitlines():
    if "vector" in l.lower() or "fadd" in l:
        print(l[:160])
