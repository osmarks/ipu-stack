#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.ipu_stack_add_f16,"ax",@progbits
	.globl ipu_stack_add_f16
	.p2align 2
	.type ipu_stack_add_f16,@function
ipu_stack_add_f16:
	.supervisor
	add $m11, $m11, -24
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m6, $m11, $m15, 3
	st32 $m7, $m11, $m15, 4
	setzi $m0, .Ladd_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size ipu_stack_add_f16, .-ipu_stack_add_f16

	.worker
	.p2align 2
.Ladd_f16_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m4, $mvertex_base, $m15, 2
	ld32 $m5, $mvertex_base, $m15, 3
	ld32 $m6, $mvertex_base, $m15, 4
	get $m7, $WSR
	and $m7, $m7, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m8, $m7, $m6
	mul $m9, $m7, $m5
	min $m0, $m7, $m6
	add $m9, $m9, $m0
	shl $m9, $m9, 2
	add $m2, $m2, $m9
	add $m3, $m3, $m9
	add $m4, $m4, $m9
	add $m5, $m5, $m8
.Ladd_f16_worker_loop:
	ld32 $a0, $m3, $m15, 0
	ld32 $a1, $m4, $m15, 0
	f16v2add $a0, $a0, $a1
	st32 $a0, $m2, $m15, 0
	add $m2, $m2, 4
	add $m3, $m3, 4
	add $m4, $m4, 4
	sub $m5, $m5, 1
	brnz $m5, .Ladd_f16_worker_loop
	exitz $m15
