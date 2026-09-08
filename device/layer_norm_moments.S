#include "arch/gc_tile_defines.h"
// m2 output moments, m3 source, m4 rows, m5 local feature width.
	.text
	.allow_optimizations
	.section .text.layer_norm_moments,"ax",@progbits
	.globl layer_norm_moments
	.p2align 2
	.type layer_norm_moments,@function
layer_norm_moments:
	.supervisor
	add $m11, $m11, -128
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	add $m0, $m11, 32
	st32 $m0, $m11, $m15, 3
	setzi $m0, __runCodelet_LayerNormMoments
.Lrow:
	setzi $m1, 0
.Lstage:
	st32 $m1, $m11, $m15, 4
	runall $m0, $m11, 0
	sync TEXCH_SYNCZONE_LOCAL
	add $m1, $m1, 1
	cmpeq $m2, $m1, 3
	brz $m2, .Lstage
	ld32 $m3, $m11, $m15, 0
	shl $m2, $m5, 1
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 0
	ld32 $m2, $m11, $m15, 1
	add $m2, $m2, 8
	st32 $m2, $m11, $m15, 1
	sub $m4, $m4, 1
	brnz $m4, .Lrow
	add $m11, $m11, 128
	br $m10
	.size layer_norm_moments, .-layer_norm_moments
