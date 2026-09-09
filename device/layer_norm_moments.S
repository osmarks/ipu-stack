#include "arch/gc_tile_defines.h"
#ifdef NORM_STORE_SUM
#define MOMENTS_SYMBOL add_layer_norm_moments
#define MOMENTS_VERTEX __runCodelet_AddLayerNormMoments
#define MOMENTS_ROWS $m6
#define MOMENTS_WIDTH $m7
#else
#define MOMENTS_SYMBOL layer_norm_moments
#define MOMENTS_VERTEX __runCodelet_LayerNormMoments
#define MOMENTS_ROWS $m4
#define MOMENTS_WIDTH $m5
#endif
// Output moments, source, [right, residual output], rows, local width.
	.text
	.allow_optimizations
	.section .text.MOMENTS_SYMBOL,"ax",@progbits
	.globl MOMENTS_SYMBOL
	.p2align 2
	.type MOMENTS_SYMBOL,@function
MOMENTS_SYMBOL:
	.supervisor
	add $m11, $m11, -128
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 MOMENTS_WIDTH, $m11, $m15, 2
	add $m0, $m11, 32
	st32 $m0, $m11, $m15, 3
	#ifdef NORM_STORE_SUM
	st32 $m4, $m11, $m15, 5
	st32 $m5, $m11, $m15, 6
#endif
	setzi $m0, MOMENTS_VERTEX
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
	shl $m2, MOMENTS_WIDTH, 1
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 0
#ifdef NORM_STORE_SUM
	ld32 $m3, $m11, $m15, 5
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 5
	ld32 $m3, $m11, $m15, 6
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 6
#endif
	ld32 $m2, $m11, $m15, 1
	add $m2, $m2, 8
	st32 $m2, $m11, $m15, 1
	sub MOMENTS_ROWS, MOMENTS_ROWS, 1
	brnz MOMENTS_ROWS, .Lrow
	add $m11, $m11, 128
	br $m10
	.size MOMENTS_SYMBOL, .-MOMENTS_SYMBOL
