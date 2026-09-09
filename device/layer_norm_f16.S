#include "arch/gc_tile_defines.h"

#ifdef NORM_WITH_ADD
#define NORM_SYMBOL add_layer_norm_f16
#define NORM_VERTEX __runCodelet_AddLayerNormF16
#define NORM_ROWS $m7
#define NORM_WIDTH $m8
#define NORM_GAMMA $m5
#define NORM_BETA $m6
#define NORM_SHIFT 1
#define NORM_STACK 160
#define NORM_SCRATCH 64
#elif defined(NORM_FP8)
#define NORM_SYMBOL layer_norm_f8
#define NORM_VERTEX __runCodelet_LayerNormF8
#define NORM_ROWS $m6
#define NORM_WIDTH $m7
#define NORM_GAMMA $m4
#define NORM_BETA $m5
#define NORM_SHIFT 0
#define NORM_STACK 160
#define NORM_SCRATCH 64
#else
#define NORM_SYMBOL layer_norm_f16
#define NORM_VERTEX __runCodelet_LayerNormF16
#define NORM_ROWS $m6
#define NORM_WIDTH $m7
#define NORM_GAMMA $m4
#define NORM_BETA $m5
#define NORM_SHIFT 0
#define NORM_STACK 128
#define NORM_SCRATCH 32
#endif

// Share each row across all six workers. Separate sum and centered-variance
// buffers avoid read/write races; local sync separates the three passes.
	.text
	.allow_optimizations
	.section .text.NORM_SYMBOL,"ax",@progbits
	.globl NORM_SYMBOL
	.p2align 2
	.type NORM_SYMBOL,@function
NORM_SYMBOL:
	.supervisor
	add $m11, $m11, -NORM_STACK
	st32 $m3, $m11, $m15, 0
#ifdef NORM_WITH_ADD
	st32 $m4, $m11, $m15, 1
#endif
	st32 NORM_GAMMA, $m11, $m15, (1 + NORM_SHIFT)
	st32 NORM_BETA, $m11, $m15, (2 + NORM_SHIFT)
	st32 $m2, $m11, $m15, (3 + NORM_SHIFT)
	st32 NORM_ROWS, $m11, $m15, (4 + NORM_SHIFT)
	st32 NORM_WIDTH, $m11, $m15, (5 + NORM_SHIFT)
	add $m0, $m11, NORM_SCRATCH
	st32 $m0, $m11, $m15, (6 + NORM_SHIFT)
#ifdef NORM_FP8
	st32 $m15, $m11, $m15, 8
	st32 $m8, $m11, $m15, 9
	st32 $m9, $m11, $m15, 10
#endif
	setzi $m0, NORM_VERTEX
.Lrow:
	setzi $m1, 0
.Lstage:
	st32 $m1, $m11, $m15, (7 + NORM_SHIFT)
	runall $m0, $m11, 0
	sync TEXCH_SYNCZONE_LOCAL
	add $m1, $m1, 1
	cmpeq $m2, $m1, 3
	brz $m2, .Lstage
	shl $m2, NORM_WIDTH, 1
	ld32 $m3, $m11, $m15, 0
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 0
#ifdef NORM_FP8
	ld32 $m3, $m11, $m15, 8
	add $m3, $m3, 1
	st32 $m3, $m11, $m15, 8
#else
	ld32 $m3, $m11, $m15, (3 + NORM_SHIFT)
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, (3 + NORM_SHIFT)
#endif
#ifdef NORM_WITH_ADD
	ld32 $m3, $m11, $m15, 1
	add $m3, $m3, $m2
	st32 $m3, $m11, $m15, 1
#endif
	sub NORM_ROWS, NORM_ROWS, 1
	brnz NORM_ROWS, .Lrow
	add $m11, $m11, NORM_STACK
	br $m10
	.size NORM_SYMBOL, .-NORM_SYMBOL
