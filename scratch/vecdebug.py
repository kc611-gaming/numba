import llvmlite.binding as llvm
llvm.set_option('', '--debug-only=loop-vectorize')

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
