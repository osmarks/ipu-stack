	.text
	.allow_optimizations

	.section .text.ipu_stack_add_u32,"ax",@progbits
	.globl ipu_stack_add_u32
	.p2align 2
	.type ipu_stack_add_u32,@function
ipu_stack_add_u32:
	.supervisor
	ld32 $m8, $m3, $m15, 0
	ld32 $m9, $m4, $m15, 0
	add $m8, $m8, $m9
	st32 $m8, $m2, $m15, 0
	br $m10
	.size ipu_stack_add_u32, .-ipu_stack_add_u32
