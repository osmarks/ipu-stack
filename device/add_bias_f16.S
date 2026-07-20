#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.ipu_stack_add_bias_f16_c16,"ax",@progbits
	.globl ipu_stack_add_bias_f16_c16
	.p2align 2
	.type ipu_stack_add_bias_f16_c16,@function
ipu_stack_add_bias_f16_c16:
	.supervisor
	add $m11, $m11, -24
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 $m7, $m11, $m15, 5
	setzi $m0, .Ladd_bias_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size ipu_stack_add_bias_f16_c16, .-ipu_stack_add_bias_f16_c16

	.worker
	.p2align 2
.Ladd_bias_worker:
	ld32 $m5, $mvertex_base, $m15, 3
	ld32 $m6, $mvertex_base, $m15, 4
	ld32 $m7, $mvertex_base, $m15, 5
	get $m8, $WSR
	and $m8, $m8, CSR_W_WSR__CTXTID_M1__MASK
	mov $m9, $m6
	cmpult $m0, $m8, $m7
	add $m9, $m9, $m0
	brz $m9, .Ladd_bias_done
	mul $m10, $m8, $m6
	min $m0, $m8, $m7
	add $m10, $m10, $m0
.Ladd_bias_row:
	zero $m0
.Ladd_bias_panel:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m4, $mvertex_base, $m15, 2
	mul $m1, $m5, 32
	mul $m1, $m1, $m0
	add $m2, $m2, $m1
	add $m3, $m3, $m1
	mul $m1, $m10, 32
	add $m2, $m2, $m1
	add $m3, $m3, $m1
	shl $m1, $m0, 5
	add $m4, $m4, $m1
	setzi $m7, 8
.Ladd_bias_word:
	ld32 $a0, $m3, $m15, 0
	ld32 $a1, $m4, $m15, 0
	f16v2add $a0, $a0, $a1
	st32 $a0, $m2, $m15, 0
	add $m2, $m2, 4
	add $m3, $m3, 4
	add $m4, $m4, 4
	sub $m7, $m7, 1
	brnz $m7, .Ladd_bias_word
	add $m0, $m0, 1
	setzi $m1, 4
	cmpult $m1, $m0, $m1
	brnz $m1, .Ladd_bias_panel
	add $m10, $m10, 1
	sub $m9, $m9, 1
	brnz $m9, .Ladd_bias_row
.Ladd_bias_done:
	exitz $m15
