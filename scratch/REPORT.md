# Numba SIMD gaps vs NumPy — investigation report

Investigated in this sandbox against an installed numba 0.67.0 / numpy 2.2.6 /
llvmlite 0.49.0 (CPU: AMD EPYC, AVX2+AVX512), using the local numba source
checkout for code references. Experiment patches were applied to the
*installed* numba (site-packages) only; the git checkout was not modified.
Backups of the two patched site-packages files are in
`../pkgcache/npyimpl.py.orig` and `../pkgcache/arraymath.py.orig`.

## 1. User reports found (GitHub / forums)

| Issue | Report |
|---|---|
| numba#10060 (open) | `np.exp` 6x slower than numpy; numpy dispatches AVX512_SKX, numba produces no SIMD |
| numba#9819 (open) | `np.sum` ~10x slower than numpy for non-C-contiguous arrays |
| numba#9679 (open) | numba `sum` speed very axis-dependent |
| numba#5373 / #9611 (open) | numba `np.sort` significantly slower than numpy |
| numba#5033 | loops via `np.nditer` / `@overload` are not SIMD-vectorized |
| numba#5198 | accumulating into an array element blocks vectorization |
| numba#2390 (open) | "better loop index codegen for SIMD vectorization" — the general form of the loop-codegen problem |
| numba#2088, #3984 | general codegen-for-SIMD and docs gaps |

Context: Intel SVML support (which previously provided vector exp/log/etc.)
was removed in numba 0.61; pip-installed numba now has no vector math library.

## 2. Local reproduction (all confirmed)

1M float64 elements, best-of-10:

| Case | numpy | numba | ratio |
|---|---|---|---|
| np.exp | 954 µs | 5154 µs | **5.4x slower** |
| np.log | 1114 µs | 4709 µs | **4.2x slower** |
| np.sum (1d C) | 224 µs | 863 µs | 3.9x slower |
| np.sum (3d transposed, #9819) | 435 µs | 4636 µs | **10.7x slower** |
| np.sum(x*x) no fastmath | 443 µs | 1115 µs | 2.5x slower |
| np.sort | 62.2 ms | 111.6 ms | 1.8x slower |
| np.cumsum | 3.64 ms | 1.05 ms | numba 3.5x faster (not a gap) |

## 3. Root causes (verified with LLVM diagnostics)

1. **Pointer-induction loops.** Numba's builtin reductions (`_numpy_sum`,
   prod/min/max via `ArrayIterator`, `numba/np/arraymath.py`) and the
   non-contiguous `_numpy_sum_axis` path emit loops that walk a *pointer*
   by a *runtime* stride. LLVM's loop vectorizer rejects them:
   `LV: Not vectorizing: Found an unidentified PHI ... Cannot prove legality`
   (exactly the problem numba#2390 has described since 2017). A plain
   `for i in range(n): acc += x[i]` loop compiles to an integer-index loop
   and vectorizes fine — so this is purely a codegen-shape issue.
2. **Ufunc element loop.** `numpy_ufunc_kernel` (`numba/np/npyimpl.py`) emits
   per-dimension index allocas with conditional updates + runtime-stride GEP
   chains; LV bails with "Early exit is not the latch predecessor" /
   "unidentified PHI". `np.exp(x)` therefore runs the scalar libm `exp`
   per element while numpy uses its hand-vectorized npyv loop.
3. **No vector math library.** When the loop *is* vectorizable, LLVM widens
   `llvm.exp.f64` to `llvm.exp.v8f64`, but the x86 backend scalarizes vector
   libm intrinsics into per-lane scalar calls — zero speedup. SVML used to
   fix this; since its removal nothing does. llvmlite 0.49 offers no way to
   set the vector library: the `-vector-library=LIBMVEC-X86` cl::opt is not
   honored through `create_pass_builder`, and attaching a VFABI
   `vector-function-abi-variant` attribute engages LV but crashes llvmlite's
   LLVM 20 (VPlan assertion `VPlanHelpers.h:240`).
4. **Sort**: numpy ≥1.25 uses the x86-simd-sort AVX512/AVX2 kernels; numba
   uses its own scalar introsort (no SIMD applicable without porting).
5. **Non-contiguous reductions**: numba iterates in index order; numpy's
   nditer iterates in memory order so its inner loop is unit-stride and
   vectorizes. A naive memory-order user loop in numba is *also* slow
   (73 ms vs numpy 0.39 ms) because generic multi-dim `x[i,j,k]` getitem
   uses runtime strides and is not recognized as unit-stride either — the
   fix must be in codegen (same family as cause 1).

## 4. Prototype fixes (all measured working)

Patched files (installed 0.67 tree; backups kept):

### A. Flat 1d fast path in `numpy_ufunc_kernel` (npyimpl.py)
For unary ufuncs over 1d contiguous arrays, emit a flat unit-stride index
loop (`gep(data, i)`) instead of the generic multi-dim machinery:
`v = kernel.generate(inp.load_data([i])); out.store_data([i], v)`.

### B. Index-loop fast path in `_numpy_sum` (arraymath.py)
For 1d C/F arrays, replace the ArrayIterator pointer loop with an
integer-index loop (`gep(ary.data, [i])`).

### C. libmvec vector math (codegen hook)
After module optimization, rewrite `llvm.{exp,log,sin,cos}.v{2,4,8}f64` /
`.v{4,8,16}f32` calls to glibc libmvec symbols (`_ZGVdN4v_exp`,
`_ZGVeN8v_exp`, ...) — the same trick numba's old SVML integration used,
but at the IR-string level, and no declaration changes are needed because
renaming the intrinsic's `declare` line supplies the libmvec declaration
with correct readnone/willreturn attrs. Verified `_ZGV` calls present in
final asm; correctness 1e-13 vs numpy for f64.

### Measured results with A+B+C

| Case | before | after | numpy |
|---|---|---|---|
| np.exp f64 fastmath | 5154 µs | **1118 µs (parity, 1.02x)** | 1095 µs |
| np.exp f32 fastmath | ~8.6x slower | **371 µs — 1.5x faster than numpy** | 557 µs |
| np.log f64 fastmath | 4709 µs | 1457 µs (1.30x) | 1117 µs |
| np.sum fastmath | 863 µs | **99 µs — 2.5x faster than numpy** | 247 µs |
| np.sum strict | 863 µs | 963 µs (unchanged semantics, sequential order kept) | 247 µs |

Validation battery (empty, len-1, F-order, 2d/3d, transposed, non-contig
column slices, int/bool sums) passed; only float32 1-ulp diffs vs numpy
(max 2.4e-7) remain, which is expected and matches the 2-ULP tolerance
numba's old SVML tests used.

## 5. Suggested improvements for numba (concrete)

1. **Unit-stride inner loops in codegen** (fixes #2390, #9819, #5033, most
   of the sum gaps):
   - `_numpy_sum`/prod/min/max/argmin/argmax and the ufunc kernel: when the
     innermost axis is contiguous, emit an integer-index flat loop
     (`gep(data, i)`) instead of pointer/stride stepping (prototyped in A/B).
   - Longer term: teach the array-access codegen to iterate in memory order
     and to prove unit stride for the inner axis so LLVM can vectorize
     arbitrary reductions and copies.
2. **Vector math via libmvec** (fixes #10060 for Linux/glibc): implement what
   prototype C does as a real pass — after the final optimization, rewrite
   `llvm.*.v*` math intrinsic calls to `_ZGV*` libmvec symbols when
   fastmath is set (errno-free), gated on a runtime check that libmvec is
   loadable (mirrors the old `_try_enable_svml` logic in numba 0.60's
   `numba/__init__.py`). Alternatively, request a `vector_library` option in
   llvmlite's target-machine/pass-builder API so the LLVM-native
   `-vector-library=LIBMVEC-X86` mechanism can be used (also fixes the
   VFABI/VPlan crash path).
3. **Pairwise summation** for `np.sum` to match numpy's accuracy *and*
   enable non-fastmath SIMD (numpy-compatible, closes most of the remaining
   strict-sum gap).
4. **Sort**: consider adopting x86-simd-sort (as numpy did) for contiguous
   int/float arrays (#5373/#9611).

## 6. Caveats

- Vector math rewrite must be gated on fastmath (libmvec functions don't
  set errno) and on libmvec availability (glibc ≥ 2.22, x86_64 Linux).
- Broadcast (length-1 input) handling needs a runtime guard in the ufunc
  fast path before it can be a real patch.
- Prototype patches live in site-packages of this ephemeral sandbox only;
  restore with the `.orig` backups if further experiments are needed.
- Per numba's AI tools policy
  (https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html),
  any contribution derived from LLM output must be attributed via the
  mechanism described there before submission upstream.
