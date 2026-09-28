"""Patch the installed numba (experiment):
A) numpy_ufunc_kernel: flat unit-stride loop fast path for 1d C arrays
B) _numpy_sum: index-loop fast path for 1d C arrays
C) libmvec rewrite of vector math intrinsics after optimization
"""
import re
import sys

SP = "/home/agent_0f395d2c-5969-4e6e-9908-6721577f996f/.local/lib/python3.10/site-packages/numba"

# ---- Patch A: npyimpl.py ----
p = f"{SP}/np/npyimpl.py"
src = open(p).read()

FAST_PATH = '''
def _flat_1d_fastpath(context, builder, sig, args, ufunc, arguments, inputs, outputs, loopshape):
    """Fast path: all operands are 1d C-contiguous arrays of the same
    length; emit a flat unit-stride index loop so the LLVM loop
    vectorizer can vectorize the whole element loop (incl. math calls).
    """
    intpty = context.get_value_type(types.intp)
    if len(loopshape) != 1:
        return False
    for a in inputs + outputs:
        if a.ndim != 1 or a.layout not in 'CF':
            return False
    n = loopshape[0]
    with builder.if_then(builder.icmp_signed('>', n, intpty(0)), likely=True):
        with cgutils.for_range(builder, n, intp=intpty) as i:
            vals_in = []
            for arg in inputs:
                vals_in.append(arg.load_data([i]))
            vals_out = _unpack_output_values(ufunc, builder, kernel.generate(*vals_in))
            for val_out, output in zip(vals_out, outputs):
                output.store_data([i], val_out)
    out = _pack_output_values(ufunc, context, builder, sig.return_type, [o.return_val for o in outputs])
    return impl_ret_new_ref(context, builder, sig.return_type, out)

'''

if "_flat_1d_fastpath" not in src:
    src = src.replace(
        "def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):",
        FASTPATH_SRC + "\ndef numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):")

# insert kernel variable + call before the loop_nest
src = src.replace(
    """    inputs = arguments[:ufunc.nin]
    outputs = arguments[ufunc.nin:]
    assert len(outputs) == ufunc.nout

    outer_sig = _ufunc_loop_sig(
        [a.base_type for a in outputs],
        [a.base_type for a in inputs]
    )
    kernel = kernel_class(context, builder, outer_sig)
    intpty = context.get_value_type(types.intp)

    indices = [inp.create_iter_indices() for inp in inputs]
""",
    '''    inputs = arguments[:ufunc.nin]
    outputs = arguments[ufunc.nin:]
    assert len(outputs) == ufunc.nout

    outer_sig = _ufunc_loop_sig(
        [a.base_type for a in outputs],
        [a.base_type for a in inputs]
    )
    kernel = kernel_class(context, builder, outer_sig)
    intpty = context.get_value_type(types.intp)

    _fast = _flat_1d_fastpath(context, builder, sig, args, ufunc,
                              kernel, inputs, outputs, outputs[0].shape)
    if _fastpath_enabled and _fastpath_ret is not None:
        return _fastpath_ret

    indices = [inp.create_iter_indices() for inp in inputs]''')

src = src.replace(
    'def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):\n',
    '''def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):
    _fastpath_ret = None
    globals()['_FASTPATH_ACTIVE'] = False
''', 1)

open(p, 'w').write(src)
print("npyimpl patched (stage 1)")
