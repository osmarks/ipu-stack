#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set RECIPROCAL_3_SHL17, 43691
	// Callable supervisor ABI: m2 output, m3 initial partial, m4 packed remote
	// partials, m5 remote-partial count, m6 physical half-element count, and
	// m10 return. Each remote partial occupies one contiguous element-count row.
	.section .text.ipu_stack_reduce_sum_f16,"ax",@progbits
	.globl ipu_stack_reduce_sum_f16
	.p2align 2
	.type ipu_stack_reduce_sum_f16,@function
ipu_stack_reduce_sum_f16:
	.supervisor
	add $m11, $m11, -24
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	setzi $m0, .Lreduce_sum_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size ipu_stack_reduce_sum_f16, .-ipu_stack_reduce_sum_f16

	.worker
	.p2align 3
.Lreduce_sum_f16_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m4, $mvertex_base, $m15, 2
	ld32 $m0, $mvertex_base, $m15, 3
	ld32 $m9, $mvertex_base, $m15, 4
	shr $m9, $m9, 3
	shl $m1, $m9, 1
	get $m5, $WSR
	and $m5, $m5, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m6, $m5, $m9
	brz $m6, .Lreduce_sum_done
	shl $m10, $m5, 4
	add $m2, $m2, $m10
	add $m3, $m3, $m10
	add $m4, $m4, $m10
	add $m6, $m9, 5
	sub $m6, $m6, $m5
	setzi $m10, RECIPROCAL_3_SHL17
	mul $m6, $m6, $m10
	shr $m6, $m6, 18
	brnzdec $m6, .Lreduce_sum_loop
	bri .Lreduce_sum_done
.Lreduce_sum_loop:
	ld64step $a0:1, $mzero, $m3+=, 1
	ld64step $a2:3, $mzero, $m3+=, 11
	mov $m7, $m4
	add $m10, $m4, 8
	mov $m8, $m0
.Lreduce_sum_partials:
	ld64step $a4:5, $mzero, $m4+=, $m1
	ld64step $a6:7, $mzero, $m10+=, $m1
	f16v4add $a0:1, $a0:1, $a4:5
	f16v4add $a2:3, $a2:3, $a6:7
	sub $m8, $m8, 1
	brnz $m8, .Lreduce_sum_partials
	add $m4, $m7, 96
	st64step $a0:1, $mzero, $m2+=, 1
	st64step $a2:3, $mzero, $m2+=, 11
	brnzdec $m6, .Lreduce_sum_loop
.Lreduce_sum_done:
	exitz $m15
