# Numba performance complaints vs alternatives — ranklist

A ranked list of problems where users reported Numba being slow compared to an
alternative, compiled from public forums, issue trackers and blogs
(september 2026).

Ranking = frequency of independent reports x reported slowdown magnitude x
user impact (workarounds needed / abandonment).

## Sources used

| Source | What was mined |
|---|---|
| GitHub `numba/numba` issue tracker (searches: "slow", "slower than", "compilation slow") | ~30 performance issues |
| Hacker News (Algolia API, comments + stories) | ~25 relevant comments/stories |
| numba.discourse.group (search "slow") | 43 matching topics |
| discourse.julialang.org (search "numba slower") | 45 topics (mostly counter-evidence) |
| Blogs: pythonspeed.com, vikramsg.github.io, stochasticlifestyle.com | 3 articles |

Blocked/unusable at research time: Stack Overflow (API throttled from this IP,
HTML behind captcha), Reddit (403 / rate-limited), Bing/DDG (junk results).
Known SO threads exist (e.g. SO 27287867 "slow numba jit time in interactive
mode", referenced from numba issue #881) but could not be verified directly.

---

## Ranklist

| # | Problem | Numba slower than | Typical reported gap | Reports |
|---|---|---|---|---|
| 1 | Typed containers: `typed.List` / `typed.Dict` / jitclass | Python list, NumPy array, reflected list | 2x – 350x | 14+ |
| 2 | Compile time / first-call JIT overhead | AOT (Cython, C), Julia precompilation | seconds to 30 min; can exceed total runtime | 15+ |
| 3 | Re-implemented NumPy functions missing SIMD / naive algorithms | NumPy | 5x – 700x | 12+ |
| 4 | JIT<->Python boundary: callbacks, dispatch, higher-order APIs | pure compiled path, SciPy, Julia, JAX | ~25 us/call dispatch; 10x–22x end-to-end | 10+ |
| 5 | NumPy-style whole-array code inside `@njit` (the "vectorized JIT" trap) | plain NumPy | ~0–10% (sometimes slower) | 6+ |
| 6 | `parallel=True` / `prange` overhead and scaling | serial Numba, best-of-N cores | 1–4x slower, scaling stops ~8 cores | 8+ |
| 7 | String/unicode operations | CPython | several x | 4 |
| 8 | Numba CUDA kernels | CUDA C, PyTorch | ~2x | 5 |
| 9 | End-to-end workflows lost to pandas/Julia -> users moved away | pandas, Julia | 1.7x – 22x | 8+ |
| 10 | Long-tail codegen quirks and regressions | various | 2x – 2,700,000x | 12+ |

---

## 1. Typed containers (`typed.List`, `typed.Dict`, jitclass)

The single most recurring complaint: Numba's memory-tracked containers are
slower than the Python objects they replace, sometimes slower than pure CPython.

- numba#4584 — Typed list of arrays 47x slower than reflected list (0.0148 s vs 0.0003 s per run). https://github.com/numba/numba/issues/4584
- numba#4364 — iterating a typed Dict ~150x slower than a structured-dtype array (1.23 ms vs 8.3 us). https://github.com/numba/numba/issues/4364
- numba#9535 — random access into a plain Python list inside njit 350x slower than pure Python; typed List still ~5x slower. https://github.com/numba/numba/issues/9535
- numba#5909 — Dijkstra with typed lists is slower than the pure-Python version (0.91 s vs 0.78 s). https://github.com/numba/numba/issues/5909
- numba#7925 — typed.List 2x slower than reflected list on Intel and AMD. https://github.com/numba/numba/issues/7925
- numba#6625 — jitclass + typed.List 14x slower than Julia struct vector (14 us vs 1.02 us). https://github.com/numba/numba/issues/6625
- numba#9519 — typed dict falls back to memcmp/memcpy for primitive keys. https://github.com/numba/numba/issues/9519
- numba#9712 — njit allocations (np.empty_like) slower and jittery vs NumPy. https://github.com/numba/numba/issues/9712
- Discourse: appending one element slower than recreating a NumPy array (t/2824); typed.List "oddly a lot slower than numpy arrays on all fronts" (t/2780); typed dicts "orders of magnitude slower" for reads/writes (t/2442); setitem still ~3x slower than ndarray after a 5x fix (t/2560); slow appends of structured arrays (t/2696); indexing typed list of functions is the bottleneck (t/2701). https://numba.discourse.group/t/2824 /t/2780 /t/2442 /t/2560 /t/2696 /t/2701

## 2. Compile time / first-call JIT overhead

The top reason users abandoned Numba on HN; the only category with reports of
compilation taking longer than the total runtime of the task.

- numba#5696 — dynamically generated filter compiles in ~100 s vs <60 s to run unjitted: "a kind of deoptimization". https://github.com/numba/numba/issues/5696
- numba#8135 — compile time quadratic in number of structured-array fields (19 s for 141 fields). https://github.com/numba/numba/issues/8135
- numba#5832 — initial compilation 5x slower (0.94 s -> 4.72 s) between releases. https://github.com/numba/numba/issues/5832
- numba#881 — 15 s JIT time in interactive mode. https://github.com/numba/numba/issues/881 (also SO 27287867)
- numba#6169 — overload_method compiles slower than jitclass for same functionality. https://github.com/numba/numba/issues/6169
- Discourse: compilation up to ~30 min for large workloads (t/3000); jitclass init takes seconds (t/2535); cached first call still ~3.7x slower (t/2783); device-function edits invalidate CUDA cache and force minute-long recompiles (t/2396). https://numba.discourse.group/t/3000 /t/2535 /t/2783 /t/2396
- HN: a team found Numba "a real pain" beyond hot loops and moved to Julia (https://news.ycombinator.com/item?id=22139159); Numba's creator notes compile times drove the move away from AST compilation (https://news.ycombinator.com/item?id=8933142).

## 3. Re-implemented NumPy functions slower than NumPy

Numba reimplements NumPy in its own language; users repeatedly find the
reimplementation loses badly (no SIMD, naive algorithms, NaN corner cases).

- numba#8382 — np.sort of an all-NaN array: 3.82 s in njit vs 4.17 ms via py_func (~900x). https://github.com/numba/numba/issues/8382
- numba#5373 — np.sort inside njit(parallel) 22x slower than NumPy (6.69 ms vs 295 us). https://github.com/numba/numba/issues/5373
- numba#9611 — jitted np.sort / np.unique >5x slower than plain NumPy. https://github.com/numba/numba/issues/9611
- numba#9679 — np.sum up to 13x slower and strongly axis-dependent (26.8 ms vs 2.06 ms). https://github.com/numba/numba/issues/9679
- numba#9819 — np.sum on non-C-contiguous arrays ~12x slower. https://github.com/numba/numba/issues/9819
- numba#10060 — njit np.exp 6x slower than NumPy (no AVX512 SIMD). https://github.com/numba/numba/issues/10060
- numba#233 (historical) — integer modulo 15x slower than C. https://github.com/numba/numba/issues/233
- Discourse: np.argmax slower than NumPy (t/2331); np.random slower for partial permutation (t/2389); np.cov first-call spike per new shape (t/2917); np.diff slow/misbehaving (t/2324); `arr[:, i]` column views much slower than NumPy (t/2462). https://numba.discourse.group/t/2331 /t/2389 /t/2917 /t/2324 /t/2462

## 4. JIT<->Python boundary overhead

Python objects crossing into jitted code (callbacks, kwargs, custom types)
force round-trips that erase the JIT advantage; the core argument in the
"packages compose badly" critiques.

- HN (Chris Rackauckas): APIs taking higher-order functions force costly round-trips, making Numba+SciPy ~10x slower than it should be. https://news.ycombinator.com/item?id=17205634
- Blog (Rackauckas): SciPy `odeint` + Numba-jitted RHS is 13x–22x slower than the equivalent Julia solution on a real Lorenz benchmark; JAX also beats it. https://www.stochasticlifestyle.com/why-numba-and-cython-are-not-substitutes-for-julia/
- Discourse: default/optional kwargs cost ~24.5 us per call (t/2309, t/3235); overload of scipy.integrate.quad slower than unjitted scipy (t/3011); custom types dispatch ~10x slower than builtins (numba#5839); gufunc calling external .so adds per-call slowness (t/2735/t/2733); PyTensor cache wrappers slow dispatch (t/3077). https://numba.discourse.group/t/2309 /t/3235 /t/3011 /t/3077

## 5. The "vectorized NumPy code in Numba" trap

Adding `@jit` to NumPy-style whole-array code yields ~no speedup because the
temporary-array architecture is unchanged; the loop rewrite is what wins.

- Blog (pythonspeed): grayscale conversion — NumPy 2712 us vs "wrong way" Numba 2446 us (~10% faster) vs loop version 536 us (5x). https://pythonspeed.com/articles/slow-numba/
- Discourse: jitted code slower than original code (t/2878); jitted cross-correlation slower than raw Python since it is memory-bandwidth bound (t/2564); jitted np.ones-allocating variant much slower than scalar loop (t/2351). https://numba.discourse.group/t/2878 /t/2564 /t/2351
- HN: square-root + float-modulo loop over coordinates — Numba "significantly slower than vanilla Python" on code that looked ideal for it. https://news.ycombinator.com/item?id=26340267

## 6. `parallel=True` / `prange` overhead and poor scaling

- numba#10641 — `prange(limit)` is ~4x slower than `prange(1, limit)` (40.6 ms vs 9.57 ms). https://github.com/numba/numba/issues/10641
- Discourse: prange 2x–3x slower than serial on 300k-node workload (t/2514); ~1% overhead even with one thread (t/3085); parallel slower for eigvalsh (t/2835) and for short compute where thread overhead dominates (t/2912); threads competing for L1 cache (t/2413); try/except loses all parallel speedup (t/3075). https://numba.discourse.group/t/2514 /t/3085 /t/2835 /t/2912 /t/3075 /t/2413
- HN: a Numba-parallel function stops scaling past ~8 cores. https://news.ycombinator.com/item?id=38697404

## 7. Strings / unicode

- numba#7535 — Numba unicode ops much slower than CPython (brute-force substring find vs CPython's bloom/KMP short-cuts). https://github.com/numba/numba/issues/7535
- Discourse: unicode implementation slower than CPython for string-heavy code (t/2557). https://numba.discourse.group/t/2557
- HN: type-unstable, IO and text-processing code stays slow. https://news.ycombinator.com/item?id=10748160

## 8. Numba CUDA vs CUDA C / PyTorch

- Discourse: kernels consistently slower than CUDA C due to emitted memory-safety checks (t/2554); slower than PyTorch from non-coalesced access (t/2404); dispatch slower without explicit signatures (t/2848). https://numba.discourse.group/t/2554 /t/2404 /t/2848
- Counter-point: other users find Numba CUDA faster than Julia CUDA on V100. https://discourse.julialang.org/t/66341

## 9. End-to-end losses -> users moved to alternatives

- Blog (vikramsg): Numba groupby-interpolate pipeline 0.59 s vs pandas 0.33 s vs NumPy rewrite 0.036 s; "so slow that I did not want to delve deeper". https://vikramsg.github.io/blog_numba_slower_than_pandas/
- Discourse: numbakit-ode slower than SciPy for fine 2D/3D discretizations (t/2925); PyTorch and CPython beat Numba in specific kernels (t/2404, t/2557).
- HN: "whenever I tried Numba I found it slower than standard Python" (https://news.ycombinator.com/item?id=8642840); "tried Numba twice on real code, both times significantly slower than straight Python/numpy" (https://news.ycombinator.com/item?id=5886149); a team migrated to Julia (https://news.ycombinator.com/item?id=22139159).

## 10. Long-tail codegen quirks and regressions

- numba#6946 — inlining an array realloc branch (or even reordering conditions) makes untouched code 10x–20x slower. https://github.com/numba/numba/issues/6946
- numba#3031 (historical) — `while` loop 2.7 million times slower than the equivalent `for` loop. https://github.com/numba/numba/issues/3031
- numba#1067 — structured-array field access 10x slower than plain 2D arrays. https://github.com/numba/numba/issues/1067
- numba#4951 — np.record passed as parameter ~3x slower than recarray. https://github.com/numba/numba/issues/4951
- numba#6975 — heappushpop batch 6.5x slower than pure Python. https://github.com/numba/numba/issues/6975
- numba#3026 — repeated cloudpickle deserialization pays ~60 ms/call (dask). https://github.com/numba/numba/issues/3026
- numba#5839 — dispatch of custom types ~10x more overhead than builtins. https://github.com/numba/numba/issues/5839
- Discourse: 0.66.0rc1 makes NuCS test suite 3x slower (t/3250); structref 2x slower than dtype version (t/3063); `forceobj` calls very slow (t/2735). https://numba.discourse.group/t/3250 /t/3063 /t/2735

---

## Caveats and counter-evidence

- Many reports are historical and since fixed (#233 modulo, #3031 while-loop,
  parts of #7535). Several remain open years later (#4584, #5373, #7925,
  #10060, #9712), i.e. still reproducible pain points.
- Numba frequently wins too: on Julia Discourse, 12 of 45 "numba slower"
  search hits actually show Numba beating Julia (ODE loops, Mandelbrot,
  CUDA matmul, trig loops); Codon and PyPy lose to Numba in other HN reports
  (https://news.ycombinator.com/item?id=41352789, 42213416).
- Several "Numba is slow" cases are measurement artifacts (JIT warm-up
  included, memory-bound workloads where no compiler can help) — respondents
  on both Discourse forums regularly point this out.
- Stack Overflow and Reddit could not be mined directly during this research
  (API throttling / bot walls); the ranklist is based on the sources listed
  above.

*This document was produced with AI assistance (Kilo, claude-based agent) from
public sources. Per the Numba AI tools policy
(https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html),
attribute appropriately if its contents feed a Numba contribution.*
