	.text
	.allow_optimizations

	.set WORKER_ARGUMENTS, 0x50000
	.set SYNC_COMPUTE_SET, 2
	.set BLOCK_DIMENSION, 64
	.set BLOCK_ROW_BYTES, BLOCK_DIMENSION * 4

	.section .text.ipu_stack_copy_4096_u32,"ax",@progbits
	.globl ipu_stack_copy_4096_u32
	.p2align 2
	.type ipu_stack_copy_4096_u32,@function
ipu_stack_copy_4096_u32:
	.supervisor
	setzi $m4, BLOCK_DIMENSION * BLOCK_DIMENSION
.Lcopy_word:
	ld32 $m0, $m3, $m15, 0
	st32 $m0, $m2, $m15, 0
	add $m3, $m3, 4
	add $m2, $m2, 4
	sub $m4, $m4, 1
	brnz $m4, .Lcopy_word
	br $m10
	.size ipu_stack_copy_4096_u32, .-ipu_stack_copy_4096_u32

	.macro GEMM_SUPERVISOR name, worker
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	setzi $m5, WORKER_ARGUMENTS
	st32 $m2, $m5, $m15, 0
	st32 $m3, $m5, $m15, 1
	st32 $m4, $m5, $m15, 2
	setzi $m0, \worker
	runall $m0, $m5, 0
	sync SYNC_COMPUTE_SET
	br $m10
	.size \name, .-\name
	.endm

	GEMM_SUPERVISOR ipu_stack_gemm_f32_64_init, .Lgemm_worker_init
	GEMM_SUPERVISOR ipu_stack_gemm_f32_64_accumulate, .Lgemm_worker_accumulate

	.worker
	.p2align 3
.Lgemm_worker_init:
	zero $m13
	bri .Lgemm_worker

	.worker
	.p2align 3
.Lgemm_worker_accumulate:
	setzi $m13, 1

.Lgemm_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m4, $mvertex_base, $m15, 2
	get $m5, $WSR
	and $m5, $m5, 7
	setzi $m0, BLOCK_DIMENSION

.Lrow:
	cmpult $m1, $m5, $m0
	brz $m1, .Lgemm_exit
	shl $m6, $m5, 8
	add $m7, $m3, $m6
	add $m11, $m2, $m6
	zero $m8

.Lcolumn:
	shl $m9, $m8, 2
	add $m9, $m4, $m9
	mov $a2, $azero
	brz $m13, .Linner_setup
	ld32 $a2, $m11, $m15, 0

.Linner_setup:
	mov $m6, $m7
	setzi $m1, BLOCK_DIMENSION
.Linner:
	ld32 $a0, $m6, $m15, 0
	ld32 $a1, $m9, $m15, 0
	f32mul $a1, $a0, $a1
	f32add $a2, $a2, $a1
	add $m6, $m6, 4
	add $m9, $m9, BLOCK_ROW_BYTES
	sub $m1, $m1, 1
	brnz $m1, .Linner
	st32 $a2, $m11, $m15, 0
	add $m11, $m11, 4
	add $m8, $m8, 1
	cmpult $m1, $m8, $m0
	brnz $m1, .Lcolumn
	add $m5, $m5, 6
	bri .Lrow

.Lgemm_exit:
	exitz $m15
