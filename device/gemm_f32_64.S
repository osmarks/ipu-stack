	.text
	.allow_optimizations

	.set WORKER_ARGUMENTS, 0x50000
	.set SYNC_COMPUTE_SET, 2
	.set BLOCK_DIMENSION, 64
	.set FP_CLR_ZAACC, 1 << 3

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
	{
		mov $m0, $m0
		setzi $a0, FP_CLR_ZAACC
	}
	{
		mov $m0, $m0
		uput $FP_CLR, $a0
	}

.Lrow:
	cmpult $m1, $m5, $m0
	brz $m1, .Lgemm_exit
	shl $m6, $m5, 8
	add $m7, $m3, $m6
	add $m11, $m2, $m6
	zero $m8

.Lcolumn:
	shl $m9, $m8, 8
	add $m9, $m4, $m9
	mov $m6, $m7
	setzi $m1, BLOCK_DIMENSION / 2
.Linner:
	ld64 $a0:1, $m6, $m15, 0
	ld64 $a2:3, $m9, $m15, 0
	f32v2mac $a0:1, $a2:3
	add $m6, $m6, 8
	add $m9, $m9, 8
	sub $m1, $m1, 1
	brnz $m1, .Linner
	f32v2gina $a4:5, $azeros, 0
	f32add $a4, $a4, $a5
	shl $m11, $m5, 8
	add $m11, $m2, $m11
	shl $m1, $m8, 2
	add $m11, $m11, $m1
	brz $m13, .Lstore
	ld32 $a6, $m11, $m15, 0
	f32add $a4, $a4, $a6
.Lstore:
	st32 $a4, $m11, $m15, 0
	add $m8, $m8, 1
	cmpult $m1, $m8, $m0
	brnz $m1, .Lcolumn
	add $m5, $m5, 6
	bri .Lrow

.Lgemm_exit:
	exitz $m15
