import llvmlite.binding as llb
llb.initialize_all_targets()
llb.initialize_all_asmprinters()
llb.set_option('', '--vector-library=LIBMVEC-X86')

# scalar libm call (not the llvm.exp intrinsic)
ir = """
define void @foo(double* %in, double* %out, i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [0, %entry], [%i.next, %loop]
  %p = getelementptr double, double* %in, i64 %i
  %v = load double, double* %p
  %r = tail call fast double @exp(double %v)
  %po = getelementptr double, double* %out, i64 %i
  store double %r, double* %po
  %i.next = add i64 %i, 1
  %c = icmp slt i64 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
declare double @exp(double) readnone nounwind willreturn
"""
mod = llb.parse_assembly(ir)
t = llb.Target.from_triple(llb.get_process_triple())
tm = t.create_target_machine(opt=2)
pmb = llb.create_pass_builder(tm, llb.create_pipeline_tuning_options(speed_level=2))
pm = pmb.getModulePassManager()
pm.run(mod, pmb)
asm = tm.emit_assembly(mod)
print("has _ZGV:", '_ZGV' in asm)
for l in asm.splitlines():
    if 'call' in l and ('exp' in l or 'ZGV' in l):
        print(l.strip())
# also verify vectorization happened
print("vector loop:", '<4 x double>' in asm or '<8 x double>' in asm)
