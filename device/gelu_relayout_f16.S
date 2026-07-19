#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set ENABLE_STOCHASTIC_ROUNDING, 1 << CSR_S_FP_ICTL__ESR__SHIFT
	.set GELU_ALPHA_F16X2, 0x3a623a62
	.set GELU_BETA_F16X2, 0x29b929b9
	.set ONE_F16X2, 0x3c003c00
	.set HALF_F16X2, 0x38003800

	.section .text.ipu_stack_gelu_f16_c16_to_a16,"ax",@progbits
	.globl ipu_stack_gelu_f16_c16_to_a16
	.p2align 2
	.type ipu_stack_gelu_f16_c16_to_a16,@function
ipu_stack_gelu_f16_c16_to_a16:
	.supervisor
	add $m11, $m11, -32
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m6, $m11, $m15, 3
	st32 $m7, $m11, $m15, 4
	get $m8, $FP_ICTL
	st32 $m8, $m11, $m15, 5
	setzi $m0, ENABLE_STOCHASTIC_ROUNDING
	or $m8, $m8, $m0
	put $FP_ICTL, $m8
	setzi $m0, .Lgelu_f16_relayout_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	ld32 $m8, $m11, $m15, 5
	put $FP_ICTL, $m8
	add $m11, $m11, 32
	br $m10
	.size ipu_stack_gelu_f16_c16_to_a16, .-ipu_stack_gelu_f16_c16_to_a16

	.macro GELU_HALF_PAIR source, destination_offset
	f16v2mul $a2, \source, \source
	f16v2mul $a2, $a2, \source
	ldconst $a4, GELU_BETA_F16X2
	f16v2mul $a2, $a4, $a2
	f16v2add $a2, \source, $a2
	ldconst $a4, GELU_ALPHA_F16X2
	f16v2mul $a2, $a4, $a2
	f16v2tanh $a2, $a2
	ldconst $a4, ONE_F16X2
	f16v2add $a2, $a4, $a2
	ldconst $a4, HALF_F16X2
	f16v2mul $a2, $a4, $a2
	f16v2mul $a2, \source, $a2
	st32 $a2, $m2, $m15, \destination_offset
	.endm

	.worker
	.p2align 3
.Lgelu_f16_relayout_worker:
	ld32 $m7, $mvertex_base, $m15, 2
	ld32 $m8, $mvertex_base, $m15, 3
	ld32 $m9, $mvertex_base, $m15, 4
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	mov $m5, $m8
	cmpult $m6, $m4, $m9
	add $m5, $m5, $m6
	mul $m6, $m4, $m8
	min $m8, $m4, $m9
	add $m6, $m6, $m8
	brz $m5, .Lgelu_f16_done

.Lgelu_f16_row:
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	mul $m8, $m7, 32
	mul $m4, $m6, 32
	add $m0, $m0, $m4
	add $m1, $m1, $m4
	setzi $m4, 4
.Lgelu_f16_column_group:
	mov $m2, $m0
	mov $m3, $m1
	ld64 $a0:1, $m3, $m15, 0
	GELU_HALF_PAIR $a0, 0
	GELU_HALF_PAIR $a1, 4
	ld64 $a0:1, $m3, $m15, 1
	GELU_HALF_PAIR $a0, 1
	GELU_HALF_PAIR $a1, 5
	ld64 $a0:1, $m3, $m15, 2
	GELU_HALF_PAIR $a0, 2
	GELU_HALF_PAIR $a1, 6
	ld64 $a0:1, $m3, $m15, 3
	GELU_HALF_PAIR $a0, 3
	GELU_HALF_PAIR $a1, 7
	add $m0, $m0, $m8
	add $m1, $m1, $m8
	sub $m4, $m4, 1
	brnz $m4, .Lgelu_f16_column_group
	add $m6, $m6, 1
	sub $m5, $m5, 1
	brnz $m5, .Lgelu_f16_row
.Lgelu_f16_done:
	exitz $m15
