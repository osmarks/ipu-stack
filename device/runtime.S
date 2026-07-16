	.text
	.allow_optimizations

	.section .text.ipu_stack_start,"ax",@progbits
	.globl ipu_stack_start
	.p2align 2
	.type ipu_stack_start,@function
ipu_stack_start:
	.supervisor
	# Participate in the application startup superstep expected by the loader.
	.long 0x41800003
	setzi $m0, ipu_stack_worker
	setzi $m1, 0x6000
	runall $m0, $m1, 1
	# A local sync resumes the supervisor only after all workers exit.
	.long 0x41800001
	setzi $m1, 0x52000
	zero $m2
	ld32 $m3, $m1, $m15, 0
	add $m2, $m2, $m3
	ld32 $m3, $m1, $m15, 1
	add $m2, $m2, $m3
	ld32 $m3, $m1, $m15, 2
	add $m2, $m2, $m3
	ld32 $m3, $m1, $m15, 3
	add $m2, $m2, $m3
	ld32 $m3, $m1, $m15, 4
	add $m2, $m2, $m3
	ld32 $m3, $m1, $m15, 5
	add $m2, $m2, $m3
	setzi $m1, 0x52020
	st32 $m2, $m1, $m15, 0
	# Let the host release the completed fixture into its diagnostic loop.
	.long 0x41800007
.Lidle:
	add $m0, $m0, 0
	bri .Lidle
	.size ipu_stack_start, .-ipu_stack_start

	.section .text.ipu_stack_worker,"ax",@progbits
	.globl ipu_stack_worker
	.p2align 2
	.type ipu_stack_worker,@function
ipu_stack_worker:
	.worker
	setzi $m0, 1
	st32 $m0, $mvertex_base, $m15, 0
	exitnz $m0
	.size ipu_stack_worker, .-ipu_stack_worker
