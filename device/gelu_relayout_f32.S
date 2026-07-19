#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set GELU_ALPHA, 0x3f4c422a
	.set GELU_BETA, 0x3d372713
	.set FLOAT_ONE, 0x3f800000
	.set FLOAT_HALF, 0x3f000000

	.section .text.ipu_stack_gelu_c16_to_a8,"ax",@progbits
	.globl ipu_stack_gelu_c16_to_a8
	.p2align 2
	.type ipu_stack_gelu_c16_to_a8,@function
ipu_stack_gelu_c16_to_a8:
	.supervisor
	add $m11, $m11, -24
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m6, $m11, $m15, 3
	st32 $m7, $m11, $m15, 4
	setzi $m0, .Lgelu_relayout_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size ipu_stack_gelu_c16_to_a8, .-ipu_stack_gelu_c16_to_a8

	.macro GELU_PAIR source_offset, destination, destination_offset
	ld64 $a0:1, $m3, $m15, \source_offset
	f32v2mul $a2:3, $a0:1, $a0:1
	f32v2mul $a2:3, $a2:3, $a0:1
	ldconst $a4, GELU_BETA
	f32v2mul $a2:3, $a4:B, $a2:3
	f32v2add $a2:3, $a0:1, $a2:3
	ldconst $a4, GELU_ALPHA
	f32v2mul $a2:3, $a4:B, $a2:3
	f32tanh $a2, $a2
	f32tanh $a3, $a3
	ldconst $a4, FLOAT_ONE
	f32v2add $a2:3, $a4:B, $a2:3
	ldconst $a4, FLOAT_HALF
	f32v2mul $a2:3, $a4:B, $a2:3
	f32v2mul $a2:3, $a0:1, $a2:3
	st64 $a2:3, \destination, $m15, \destination_offset
	.endm

	.worker
	.p2align 3
.Lgelu_relayout_worker:
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
	brz $m5, .Lgelu_relayout_done

.Lgelu_relayout_row:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	mul $m8, $m7, 64
	mul $m9, $m7, 32
	mul $m4, $m6, 64
	add $m3, $m3, $m4
	mul $m4, $m6, 32
	add $m2, $m2, $m4
	setzi $m4, 4
.Lgelu_relayout_column_group:
	add $m10, $m2, $m9
	GELU_PAIR 0,  $m2,  0
	GELU_PAIR 1,  $m10, 0
	GELU_PAIR 2,  $m2,  1
	GELU_PAIR 3,  $m10, 1
	GELU_PAIR 4,  $m2,  2
	GELU_PAIR 5,  $m10, 2
	GELU_PAIR 6,  $m2,  3
	GELU_PAIR 7,  $m10, 3
	add $m3, $m3, $m8
	add $m2, $m2, $m8
	sub $m4, $m4, 1
	brnz $m4, .Lgelu_relayout_column_group
	add $m6, $m6, 1
	sub $m5, $m5, 1
	brnz $m5, .Lgelu_relayout_row
.Lgelu_relayout_done:
	exitz $m15
