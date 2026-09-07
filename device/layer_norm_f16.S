#include "arch/gc_tile_defines.h"

// Share each row across all six workers. Separate sum and centered-variance
// buffers avoid read/write races; local sync separates the three passes.
	.text
	.allow_optimizations
	.section .text.layer_norm_f16,"ax",@progbits
	.globl layer_norm_f16
	.p2align 2
	.type layer_norm_f16,@function
layer_norm_f16:
	.supervisor
	add $m11, $m11, -128
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m2, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 $m7, $m11, $m15, 5
	add $m0, $m11, 32
	st32 $m0, $m11, $m15, 6
	setzi $m0, __runCodelet_LayerNormF16
.Lrow:
	setzi $m1, 0
.Lstage:
	st32 $m1, $m11, $m15, 7
	runall $m0, $m11, 0
	sync TEXCH_SYNCZONE_LOCAL
	add $m1, $m1, 1
	cmpeq $m2, $m1, 3
	brz $m2, .Lstage
	shl $m2, $m7, 1
	ld32 $m3, $m11, $m15, 0
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 0
	ld32 $m3, $m11, $m15, 3
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 3
	sub $m6, $m6, 1
	brnz $m6, .Lrow
	add $m11, $m11, 128
	br $m10
	.size layer_norm_f16, .-layer_norm_f16
