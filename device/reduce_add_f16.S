#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	// Callable supervisor ABI: m2 output, m3 left, m4 right, m5 physical
	// half-element count, and m10 return. Output may alias the left input.

	.section .text.ipu_stack_reduce_add_f16,"ax",@progbits
	.globl ipu_stack_reduce_add_f16
	.p2align 2
	.type ipu_stack_reduce_add_f16,@function
ipu_stack_reduce_add_f16:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	setzi $m0, .Lreduce_add_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size ipu_stack_reduce_add_f16, .-ipu_stack_reduce_add_f16

	.worker
	.p2align 3
.Lreduce_add_f16_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m4, $mvertex_base, $m15, 2
	ld32 $m8, $mvertex_base, $m15, 3
	shr $m8, $m8, 3
	get $m5, $WSR
	and $m5, $m5, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m6, $m5, $m8
	brz $m6, .Lreduce_add_f16_done
	shl $m7, $m5, 4
	add $m2, $m2, $m7
	add $m3, $m3, $m7
	add $m4, $m4, $m7
.Lreduce_add_f16_loop:
	ld128 $a0:3, $m3, $m15, 0
	ld128 $a4:7, $m4, $m15, 0
	f16v2add $a0, $a0, $a4
	f16v2add $a1, $a1, $a5
	f16v2add $a2, $a2, $a6
	f16v2add $a3, $a3, $a7
	st64 $a0:1, $m2, $m15, 0
	st64 $a2:3, $m2, $m15, 1
	add $m2, $m2, 96
	add $m3, $m3, 96
	add $m4, $m4, 96
	add $m5, $m5, 6
	cmpult $m6, $m5, $m8
	brnz $m6, .Lreduce_add_f16_loop
.Lreduce_add_f16_done:
	exitz $m15
