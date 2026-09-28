"""Patch installed numba for the SIMD-gap experiment.

A) npyimpl.numpy_ufunc_kernel: flat unit-stride fast path for unary
   ufuncs over 1d contiguous arrays (np.exp/log/sin/cos/...).
B) arraymath._numpy_sum: index-loop fast path for 1d C arrays.
"""
SP = "/home/agent_0f395d2c-5969-4e6e-9908-6721577f996f/.local/lib/python3.10/site-packages/numba"

# ---------------- A: npyimpl ----------------
p = f"{SP}/np/npyimpl.py"
src = open(p).read()

FASTPATH_FN = '''
def _flat_1d_fastpath(context, builder, sig, args, ufunc, kernel, inputs, outputs, loopshape):
    """Flat unit-stride element loop for unary ufuncs over 1d contiguous
    arrays.  Index-based addressing lets the LLVM loop vectorizer
    vectorize the loop, including mapped vector math calls.
    """
    if ufunc.nin != 1 or ufunc.nout != 1:
        return None
    inp = inputs[0]
    out = outputs[0]
    if inp.ndim != 1 or out.ndim != 1:
        return None
    if inp.layout not in 'CF' or out.layout not in 'CF':
        return None
    intpty = context.get_value_type(types.intp)
    n = loopshape[0]
    with cgutils.for_range(builder, n, intp=intpty) as i:
        v = kernel.generate(inp.load_data([i]))
        out.store_data([i], v)
    res = impl_ret_new_ref(context, builder, sig.return_type, out.return_val)
    return res

'''

anchor = "def numpy_ufunc_kernel(context, builder, sig, args, ufunc, kernel_class):"
assert anchor in src, "anchor missing"
src = src.replace(anchor, FASTPATH_FN + anchor, 1)

old = """    # assume outputs are all the same size, which numpy requires

    loopshape = outputs[0].shape
"""
new = """    # assume outputs are all the same size, which numpy requires

    loopshape = outputs[0].shape

    _fastpath_ret = _flat_1d_fastpath(context, builder, sig, args, ufunc,
                                      kernel, inputs, outputs, loopshape)
    if _fastpath_ret is not None:
        return _fastpath_ret
"""
assert old in src, "loopshape anchor missing"
src = src.replace(old, new, 1)
open(p, 'w').write(src)
print("A: npyimpl.py patched")

# ---------------- B: arraymath._numpy_sum ----------------
p2 = f"{SP}/np/arraymath.py"
src2 = open(p2).read()

old_code = """        result = cgutils.alloca_once_value(builder, zero)

        fnty = context.typing_context.resolve_value_type(operator.iadd)
        fn_sig = fnty.get_call_type(
            context.typing_context,
            (sig.return_type, sig.return_type),
            {}
        )
        if isinstance(ret_dtype, types.Boolean):
            add_funcfn = lambda builder, args: builder.or_(*args)
        else:
            add_funcfn = context.get_function(fnty, fn_sig)

        # Loop on source and copy to destination
        with ArrayIterator(context, builder, aryty, ary) as iter_val_ptr:
            val = load_item(context, builder, aryty, iter_val_ptr)
            res_val = add_funcfn(builder, (
                builder.load(result),
                context.cast(builder, val, aryty.dtype, sig.return_type)))
            builder.store(res_val, result)

        return impl_ret_borrowed(
            context, builder, sig.return_type, builder.load(result)
        )"""

new_code = """    def codegen(context, builder, sig, args):
        ary, _, _ = args

        ary = make_array(aryty)(context, builder, ary)
        if isinstance(ret_dtype, types.NPTimedelta):
            zero = context.get_constant(ret_dtype, ret_dtype(0))
        else:
            zero = context.get_constant(ret_dtype, 0)
        result = cgutils.alloca_once_value(builder, zero)

        # Flat unit-stride fast path for 1d contiguous arrays: use an
        # integer-index loop so the LLVM vectorizer can vectorize the
        # reduction instead of rejecting the pointer-induction loop.
        if (aryty.ndim == 1 and aryty.layout in 'CF'
                and not isinstance(ret_dtype, types.NPTimedelta)):
            fnty = context.typing_context.resolve_value_type(operator.iadd)
            fn_sig = fnty.get_call_type(
                context.typing_context,
                (sig.return_type, sig.return_type),
                {}
            )
            intpty = context.get_value_type(types.intp)
            if isinstance(ret_dtype, types.Boolean):
                add_funcfn = lambda builder, args: builder.or_(*args)
            else:
                add_funcfn = context.get_function(fnty, fn_sig)
            with cgutils.for_range(builder, ary.nitems, intp=intpty) as i:
                ptr = builder.gep(ary.data, [i], inbounds=True)
                val = load_item(context, builder, aryty, ptr)
                res_val = add_funcfn(builder, (
                    builder.load(result),
                    context.cast(builder, val, aryty.dtype, sig.return_type)))
                builder.store(res_val, result)
        else:
            fnty = context.typing_context.resolve_value_type(operator.iadd)
            fn_sig = fnty.get_call_type(
                context.typing_context,
                (sig.return_type, sig.return_type),
                {}
            )
            if isinstance(ret_dtype, types.Boolean):
                add_funcfn = lambda builder, args: builder.or_(*args)
            else:
                add_funcfn = context.get_function(fnty, fn_sig)

            # Loop on source and copy to destination
            with ArrayIterator(context, builder, aryty, ary) as iter_val_ptr:
                val = load_item(context, builder, aryty, iter_val_ptr)
                res_val = add_funcfn(builder, (
                    builder.load(result),
                    context.cast(builder, val, aryty.dtype, sig.return_type)))
                builder.store(res_val, result)

        return impl_ret_borrowed(
            context, builder, sig.return_type, builder.load(result)
        )"""

old_full = '''    def codegen(context, builder, sig, args):
        ary, _, _ = args

        ary = make_array(aryty)(context, builder, ary)
        if isinstance(ret_dtype, types.NPTimedelta):
            zero = context.get_constant(ret_dtype, ret_dtype(0))
        else:
            zero = context.get_constant(ret_dtype, 0)
        result = cgutils.alloca_once_value(builder, zero)

        fnty = context.typing_context.resolve_value_type(operator.iadd)
        fn_sig = fnty.get_call_type(
            context.typing_context,
            (sig.return_type, sig.return_type),
            {}
        )
        if isinstance(ret_dtype, types.Boolean):
            add_funcfn = lambda builder, args: builder.or_(*args)
        else:
            add_funcfn = context.get_function(fnty, fn_sig)

        # Loop on source and copy to destination
        with ArrayIterator(context, builder, aryty, ary) as iter_val_ptr:
            val = load_item(context, builder, aryty, iter_val_ptr)
            res_val = add_funcfn(builder, (
                builder.load(result),
                context.cast(builder, val, aryty.dtype, sig.return_type)))
            builder.store(res_val, result)

        return impl_ret_borrowed(
            context, builder, sig.return_type, builder.load(result)
        )'''

if old_full in src2:
    src2 = src2.replace(old_full, new_code, 1)
    open(p2, 'w').write(src2)
    print("B: arraymath patched")
else:
    print("B: pattern NOT found")
