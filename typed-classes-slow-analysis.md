# Why Numba's typed classes are slow — source-level analysis

Investigation of the "typed containers and jitclasses" complaint cluster from
the ranklist (`numba-slow-vs-alternatives-ranklist.md`), done against the
Numba source tree in this workspace. All file:line references are to this
checkout. Evidence numbers come from the linked issues/threads.

## Scope and method

Static source analysis of the three typed structures users most often report
as slow, compared against their Numba-native counterparts. Benchmarks could
not be re-run here (no numpy/numba runtime in this environment), so the
analysis maps each reported slowdown to concrete code paths, and cites the
measured numbers from the original reports.

| Typed construct | Compared against | Reported gap |
|---|---|---|
| `typed.List` | reflected list, plain Python list, NumPy array | 2x – 350x |
| `typed.Dict` | NumPy structured-dtype array, CPython dict, plain dict-in-njit patterns | up to ~150x |
| `jitclass` | Julia struct vector, AoS NumPy arrays, plain njit code | ~14x, seconds of init |
| `typed.List` of `jitclass` | Julia `Vector{struct}` | 14x (numba#6625) |

---

## 1. `typed.List` vs reflected list

### Where the code paths differ

**Reflected list** (`numba/cpython/listobj.py`) is implemented *entirely in
compiler IR* against an inline struct (`{items ptr, size, allocated, dirty}`)
holding PyObjects:

- `getitem` = fix index + guard + **raw GEP + load**, result borrowed
  (listobj.py:510-519, getitem at listobj.py:63-66).
- `iternext` yields the borrowed raw item (listobj.py:495-507).
- `setitem` does inline incref/decref in IR — no function-pointer table.

**Typed list** (`numba/typed/listobject.py` + `numba/cext/listobject.c`) is a
C container (`NB_List` with a status-code C ABI) plus compiler-level wrappers.
The C code itself is algorithmically identical to CPython's list (same growth
pattern, listobject.c:374-417); the overhead is all in the wrapper layers:

1. **Every `l[i]` is wrapped in an `Optional`.** `_gen_getitem` returns a
   tuple `(status, Optional(item))` (listobject.py:689-748): the item is
   loaded, put into an optional struct (tag + payload, alloca + store), the
   caller then unwraps it with `_nonoptional` (typedobjectutils.py:92-104).
   The optional exists only to represent "index not found" — but
   `handle_index` (listobject.py:651-663) has *already* validated the index
   and raised before `_list_getitem` is called, so the optional is pure
   legacy overhead on every single access. `getitem_unchecked`
   (listobject.py:1299-1309) still pays for it.
2. **Refcounting goes through a vtable.** Item incref/decref are function
   pointers stored in `NB_List.methods` (listobject.c:99-115), called
   indirectly per element op (setitem: decref+copy+incref,
   listobject.c:286-303; append: incref, listobject.c:331-347). The wrappers
   are generated per type (typedobjectutils.py:123-150) but reached through
   an opaque pointer at runtime, so LLVM cannot inline them, cannot hoist the
   refcount work out of loops, and the memcpy/setitem C call is opaque.
   The reflected list does all of this inline in IR.
3. **Double bounds-checking.** The compiler level checks the index
   (`handle_index` → `fix_index` → len load + sext/trunc + select + range
   check + exception raise path), then C `numba_list_setitem`/`delitem`
   re-checks `valid_index` (listobject.c:292-296, with FIXME comments
   acknowledging the duplication).
4. **Status-code plumbing.** Every op returns `int32` status that the
   overload must branch on (impl_append listobject.py:592-621, setitem
   listobject.py:829-850), adding a branch cascade on the hot path.
5. **Iteration** re-enters the full getitem machinery per element
   (borrowed variant, still optional-wrapped) plus a mutation check
   (listobject.py:1511-1538); the C iterator already hands back a raw
   pointer (listobject.c:593-607) that is then wrapped and unwrapped.
6. **Python-side `List` wrapper pays one full JIT dispatcher call per
   operation.** `numba/typed/typedlist.py:80-142` defines `_append`,
   `_getitem`, `_setitem`, `_pop`, ... as separate `@njit` functions and the
   Python `List` class calls them one operation at a time. Each op from
   pure Python costs: Python call frame → dispatcher entry → unbox args →
   compiled op → box result ≈ ~1 µs, vs ~50 ns for `list.append`. This
   explains numba#5909's "conversion to typedlist: 1.67 s" for ~1.75 M
   elements and the Discourse report that appending one element is slower
   than recreating a NumPy array (t/2824).
7. **Nested lists multiply everything.** `ListType[ListType[int64]]` makes
   each inner list its own `NB_List` malloc + NRT meminfo + items buffer
   (listobject.c:163-196); each access to an inner list pointer-chases
   `outer struct → items → inner {meminfo,data} → inner items`.

### Mapping to reported numbers

- numba#4584 (47x, mutation of list of arrays): per access — optional wrap +
  incref + status tuple, opaque C calls and indirect refcount calls that
  block LLVM from hoisting loop-invariant array pointers, and repeated
  `meminfo_data`/base-ptr chains. Reflected list keeps the raw PyArrayObject
  pointer in a register-friendly form.
- numba#7925 (2x, Intel+AMD), t/2560 (setitem ~3x slower than ndarray even
  after a 5x fix): constant-factor sum of items 1-4 above.
- numba#9535 (350x random access into a Python list *inside* njit; typed
  List still ~5x slower): element type is unicode — every access also pays
  refcounting plus the optional machinery on top of CPython's optimised
  string-object access in pure Python.
- numba#9712 / t/2780 ("typed.List slower than numpy on all fronts"):
  creation/append go through `numba_list_new` + `numba_list_resize` (malloc +
  realloc in C) while NumPy array creation is a single amortised allocation
  with no per-element refcounting for numeric dtypes.

## 2. `typed.Dict` vs CPython dict / NumPy structured array

The C layer (`numba/cext/dictobject.c`) is a faithful port of CPython 3.7's
compact ordered dict, so asymptotics match CPython. The gaps again live in
the layers around it:

1. **Optional-wrapped lookups.** `_dict_lookup` returns `(index,
   Optional(value))`, building an optional struct + incref per hit
   (dictobject.py:430-480); `impl_getitem` then unwraps with `_nonoptional`
   (dictobject.py:767-784). CPython returns the raw pointer.
2. **Generic hash + equality with indirect calls.** Every `d[k]`/`d[k]=v`
   calls the `hash()` overload (dictobject.py:714-717, 774-776) and the C
   `key_equal` falls back to `memcmp` for primitive keys — the specialised
   equality is only installed when the key type contains NRT meminfo
   (dictobject.c:444-450; dictobject.py:290-314). This is numba#9519 (5-10%
   measured win from specialising int/float keys).
3. **Insert path allocas and memcpys.** `_dict_insert` allocates key, value
   and unused `oldval` scratch, memsets padding, and passes bytes to C
   (dictobject.py:335-378); C does lookup, entry copy (memcpy), and vtable
   increfs (dictobject.c:705-743). CPython's equivalent specialises int keys
   with inline comparisons and no byte copying.
4. **Pointer chasing.** Lookup is indices → entry → key/value with a
   size-dependent index type switch per probe (dictobject.c:299-357);
   numba#4364's 150x-vs-structured-array iteration figure reflects this plus
   per-element optional/iter machinery. (A structured-dtype NumPy array is a
   flat POD buffer with no hashing at all.)
5. **Scratch in every op**: `ptr_oldval` alloca per insert
   (dictobject.py:362-363), `memset_padding` of the key slot, status branches
   with exception paths (dictobject.py:714-727).

## 3. `jitclass` vs plain structs / NumPy arrays / Julia structs

Inside nopython code, jitclass attribute access is direct struct field access
— that part is fast. The reported slowness comes from three other places:

1. **Python-boundary boxing on every operation.** Attribute getters/setters
   and every method call from Python go through *generated `@njit` wrapper
   functions*: `accessor(self) -> self.field` (boxing.py:17-63), each one a
   full dispatcher call that boxes the result (and unboxes arguments for
   methods). Reading one float attribute from Python is a compiled-function
   call + PyObject creation, ~µs scale. Method calls additionally re-enter
   typing/dispatch machinery each call (`_generate_method`).
2. **Construction cost.** `ctor_impl` allocates an NRT meminfo with a
   registered dtor, then null-initialises the *entire* data struct before
   `__init__` runs (jitclass/base.py:556-581) — O(fields) work + allocation
   + dtor registration per instance. Building thousands of instances from
   Python (numba#6625's 1000 Points) pays this plus boxing per instance.
3. **Instances inside `typed.List` compound with problem 1.** Each element
   stored in a typed list is a `{meminfo, data}` pair pointing at its own
   separately-allocated instance: per-element access = optional wrap +
   vtable incref/decref + two-level pointer chase, versus Julia's
   `Vector{Point}` where the struct lives inline in one contiguous
   unrefcounted allocation. That is the 14 us vs 1.02 us in numba#6625.
4. **Dispatch without precompilation.** Methods are `Dispatcher` members
   typed lazily; the first call for each signature compiles (t/2535 "jitclass
   init very slow"), and from Python every call re-enters dispatcher
   machinery (default-argument dispatch overhead measured at ~24.5 us,
   t/2309).

---

## Suggested improvements (prioritised)

### typed.List / typed.Dict — biggest constant-factor wins

1. **Remove the `Optional` from `__getitem__`/`d[k]` paths.** Keep optional
   only for `get()`/`pop(default)`. Return the raw borrowed item and let the
   existing index validation raise. Touches `_gen_getitem`
   (listobject.py:689-748), `ol_getitem_unchecked`, `_dict_lookup`
   (dictobject.py:430-480) and their call sites. Highest expected win per
   line of code; directly attacks every access in #7925, t/2560, #6625.
2. **Check bounds once.** Drop the C-side re-validation (listobject.c
   FIXMEs at 293, 315, 435) or lower get/setitem fully at compiler level
   after `handle_index`, mirroring the reflected list.
3. **Inline the refcounting.** Replace the runtime vtable
   (`item_incref/item_decref`, `key_equal`, `dk_incref_*`) with direct calls
   to the already-generated per-type functions (typedobjectutils.py:123-150)
   at lowering time, or emit setitem/append inline in IR like
   `ListInstance.setitem` in listobj.py:522-529. This re-enables LICM and
   inlining of NRT calls in loops — the main blocker for #4584-class cases.
4. **Specialise primitive keys in dict.** Install direct `==` comparators
   and an identity/hash-inline path for int/float keys (fixes #9519); keep
   the C generic path for exotic keys.
5. **Fast interpreter-side ops.** Implement `List.append`/`__getitem__`/
   `__setitem__`/`Dict.__setitem__` against `NB_List`/`NB_Dict` via a small
   C extension (CPython C-API) instead of one `@njit` dispatcher call per op
   (typedlist.py:80-142). Add a bulk `List(iterable)` constructor that
   appends in one compiled call — removes ~1 µs/element boundary tax
   (numba#5909, t/2824).
6. **Iterator yields raw items.** Lower `iternext` to load from
   `numba_list_iter_next`'s pointer directly (listobject.c:593-607) without
   the (status, optional) tuple round-trip.
7. **In-place sort for numeric dtypes** using the same algorithm as
   `numpy.sort` (currently `sort()` round-trips through extra typed lists
   and arg-sort indirection, listobject.py:1257-1296).

### jitclass

8. **Cache-bound accessors.** Pre-compile getters/setters/methods for the
   declared spec signature at class definition and call the compiled entry
   points directly from Python, bypassing repeated dispatcher entry; or
   generate a `tp_getattro` on the Box subclass.
9. **Bridge to NumPy.** Auto-generate a structured dtype from the jitclass
   `spec` with box/unbox to/from an AoS array, so loops over many instances
   can run over a flat buffer (SIMD-able, no per-element meminfo). This is
   the fix path for the 14x gap in #6625 and t/3063.
10. **Skip redundant null-init** of instance data when `__init__` assigns
    all fields (base.py:576-578).

### Structural / documentation

11. Offer an explicit "unchecked" fast-path set (`getitem_unchecked`
    already exists; extend to setitem/contains) and document that typed
    containers are for *storage and crossing the JIT boundary*, not for
    per-element access in hot loops where arrays win by design (no per-item
    refcounting, flat memory, no optional/status plumbing).

## Expected impact ranking

1→2→3 attack the per-access overhead that dominates #7925/#4584/t/2560
(constant ~2-50x); 5 removes the ~1 µs/op interpreter tax seen in #5909 and
t/2824/t/2560 (up to 100x on bulk Python-side use); 4 recovers the measured
5-10% on dict-heavy code (#9519); 9 targets the 14x jitclass-in-list gap
(#6625).

---

## References

Source: numba/typed/listobject.py, numba/cext/listobject.c,
numba/typed/dictobject.py, numba/cext/dictobject.c,
numba/typed/typedobjectutils.py, numba/typed/typedlist.py,
numba/typed/typeddict.py, numba/cpython/listobj.py,
numba/experimental/jitclass/{base,boxing}.py in this workspace.

Reports: numba#4584, #4364, #5909, #6625, #7925, #9519, #9535, #9712,
numba.discourse.group t/2560, t/2824, t/2780, t/2442, t/2535, t/3063.

*This document was produced with AI assistance (Kilo agent) via static
source analysis. Per the Numba AI tools policy
(https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html),
attribute appropriately if its contents feed a Numba contribution.*
