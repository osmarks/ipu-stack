#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	// Callable supervisor ABI: m2 output, m3.. inputs, then the physical
	// half-element count and m10 return. Output may alias the first input.

	.macro REDUCE_SUM_ENTRY symbol, worker, count_register
	.section .text.\symbol,"ax",@progbits
	.globl \symbol
	.p2align 2
	.type \symbol,@function
\symbol:
	.supervisor
	add $m11, $m11, -32
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 \count_register, $m11, $m15, 5
	setzi $m0, \worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 32
	br $m10
	.size \symbol, .-\symbol
	.endm

	REDUCE_SUM_ENTRY ipu_stack_reduce_sum_2_f16, .Lreduce_sum_2_f16_worker, $m5
	REDUCE_SUM_ENTRY ipu_stack_reduce_sum_3_f16, .Lreduce_sum_3_f16_worker, $m6
	REDUCE_SUM_ENTRY ipu_stack_reduce_sum_4_f16, .Lreduce_sum_4_f16_worker, $m7

	.worker
	.macro REDUCE_SUM_WORKER symbol, inputs
	.p2align 3
\symbol:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m4, $mvertex_base, $m15, 2
	.if \inputs >= 3
	ld32 $m7, $mvertex_base, $m15, 3
	.endif
	.if \inputs >= 4
	ld32 $m8, $mvertex_base, $m15, 4
	.endif
	ld32 $m9, $mvertex_base, $m15, 5
	shr $m9, $m9, 3
	get $m5, $WSR
	and $m5, $m5, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m6, $m5, $m9
	brz $m6, .Lreduce_sum_done_\@
	shl $m10, $m5, 4
	add $m2, $m2, $m10
	add $m3, $m3, $m10
	add $m4, $m4, $m10
	.if \inputs >= 3
	add $m7, $m7, $m10
	.endif
	.if \inputs >= 4
	add $m8, $m8, $m10
	.endif
.Lreduce_sum_loop_\@:
	ld128 $a0:3, $m3, $m15, 0
	ld128 $a4:7, $m4, $m15, 0
	f16v4add $a0:1, $a0:1, $a4:5
	f16v4add $a2:3, $a2:3, $a6:7
	.if \inputs >= 3
	ld128 $a4:7, $m7, $m15, 0
	f16v4add $a0:1, $a0:1, $a4:5
	f16v4add $a2:3, $a2:3, $a6:7
	.endif
	.if \inputs >= 4
	ld128 $a4:7, $m8, $m15, 0
	f16v4add $a0:1, $a0:1, $a4:5
	f16v4add $a2:3, $a2:3, $a6:7
	.endif
	st64 $a0:1, $m2, $m15, 0
	st64 $a2:3, $m2, $m15, 1
	add $m2, $m2, 96
	add $m3, $m3, 96
	add $m4, $m4, 96
	.if \inputs >= 3
	add $m7, $m7, 96
	.endif
	.if \inputs >= 4
	add $m8, $m8, 96
	.endif
	add $m5, $m5, 6
	cmpult $m6, $m5, $m9
	brnz $m6, .Lreduce_sum_loop_\@
.Lreduce_sum_done_\@:
	exitz $m15
	.endm

	REDUCE_SUM_WORKER .Lreduce_sum_2_f16_worker, 2
	REDUCE_SUM_WORKER .Lreduce_sum_3_f16_worker, 3
	REDUCE_SUM_WORKER .Lreduce_sum_4_f16_worker, 4
