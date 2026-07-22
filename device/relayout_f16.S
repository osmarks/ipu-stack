#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.ipu_stack_relayout_f16_c16_to_a16,"ax",@progbits
	.globl ipu_stack_relayout_f16_c16_to_a16
	.p2align 2
	.type ipu_stack_relayout_f16_c16_to_a16,@function
ipu_stack_relayout_f16_c16_to_a16:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m6, $m11, $m15, 3
	setzi $m0, .Lrelayout_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size ipu_stack_relayout_f16_c16_to_a16, .-ipu_stack_relayout_f16_c16_to_a16

	.worker
	.p2align 3
.Lrelayout_worker:
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	ld32 $m7, $mvertex_base, $m15, 2
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
.Lrelayout_row:
	cmpult $m5, $m4, $m7
	brz $m5, .Lrelayout_done
	mul $m6, $m4, 32
	add $m2, $m0, $m6
	add $m3, $m1, $m6
	mul $m8, $m7, 32
	ld32 $m9, $mvertex_base, $m15, 3
.Lrelayout_column_group:
	ld64 $a0:1, $m3, $m15, 0
	st32 $a0, $m2, $m15, 0
	st32 $a1, $m2, $m15, 4
	ld64 $a0:1, $m3, $m15, 1
	st32 $a0, $m2, $m15, 1
	st32 $a1, $m2, $m15, 5
	ld64 $a0:1, $m3, $m15, 2
	st32 $a0, $m2, $m15, 2
	st32 $a1, $m2, $m15, 6
	ld64 $a0:1, $m3, $m15, 3
	st32 $a0, $m2, $m15, 3
	st32 $a1, $m2, $m15, 7
	add $m2, $m2, $m8
	add $m3, $m3, $m8
	sub $m9, $m9, 1
	brnz $m9, .Lrelayout_column_group
	add $m4, $m4, 6
	setzi $m6, .Lrelayout_row
	br $m6
.Lrelayout_done:
	exitz $m15
