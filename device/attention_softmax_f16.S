#include "arch/gc_tile_defines.h"

// AMP-left panel order; FP16 probabilities with FP32 maxima and sums.
// m2 weights, m3 scores, m4 query rows, m5 valid keys.
#if ATTENTION_FULL_BLOCK
#define SOFTMAX_FRAME_BYTES 12
#else
#define SOFTMAX_FRAME_BYTES 16
#endif
	.section .text.ATTENTION_SOFTMAX_SYMBOL,"ax",@progbits
	.globl ATTENTION_SOFTMAX_SYMBOL
	.p2align 2
	.type ATTENTION_SOFTMAX_SYMBOL,@function
ATTENTION_SOFTMAX_SYMBOL:
	.supervisor
	add $m11, $m11, -SOFTMAX_FRAME_BYTES
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
#if !ATTENTION_FULL_BLOCK
	st32 $m5, $m11, $m15, 3
#endif
	setzi $m0, .Lsoftmax_worker
	runall $m0, $m11, 0
	sync TEXCH_SYNCZONE_LOCAL
	add $m11, $m11, SOFTMAX_FRAME_BYTES
	br $m10
	.size ATTENTION_SOFTMAX_SYMBOL, .-ATTENTION_SOFTMAX_SYMBOL

	.worker
	.p2align 3
.Lsoftmax_worker:
	ld32 $m5, $mvertex_base, $m15, 2
#if !ATTENTION_FULL_BLOCK
	ld32 $m10, $mvertex_base, $m15, 3
#endif
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m0, $m4, $m5
	brz $m0, .Lsoftmax_done
	add $m8, $m5, -1
	shl $m8, $m8, 5
.Lsoftmax_row:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	shl $m0, $m4, 5
	add $m3, $m3, $m0
	add $m2, $m2, $m0
	mov $m9, $m3
	// The maximum of finite FP16 inputs is exactly representable in FP16.
	ldconst $a6, 0xfbfffbff
#if ATTENTION_FULL_BLOCK
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16 - 1
#else
	shr $m7, $m10, 4
	brz $m7, .Lsoftmax_max_tail
	sub $m7, $m7, 1
#endif
.Lsoftmax_max_panel:
	ld32step $a0, $mzero, $m3+=, 1
	.rept 7
	{ ld32step $a0, $mzero, $m3+=, 1
	  f16v2max $a6, $a6, $a0 }
	.endr
	f16v2max $a6, $a6, $a0
	add $m3, $m3, $m8
	brnzdec $m7, .Lsoftmax_max_panel
#if !ATTENTION_FULL_BLOCK
.Lsoftmax_max_tail:
	and $m7, $m10, 15
	shr $m7, $m7, 1
	brz $m7, .Lsoftmax_max_scalar
	sub $m7, $m7, 1
.Lsoftmax_max_pair:
	ld32step $a0, $mzero, $m3+=, 1
	f16v2max $a6, $a6, $a0
	brnzdec $m7, .Lsoftmax_max_pair
.Lsoftmax_max_scalar:
#endif
	f16v2tof32 $a6:7, $a6
	f32max $a6, $a6, $a7
#if !ATTENTION_FULL_BLOCK
	and $m0, $m10, 1
	brz $m0, .Lsoftmax_max_ready
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32max $a6, $a6, $a0
.Lsoftmax_max_ready:
#endif
	ldconst $a4, ATTENTION_SCALE_BITS
	f32mul $a7, $a6, $a4
	// Persistent maxima/denominators follow the packed weight matrix.
	ld32 $m6, $mvertex_base, $m15, 0
	setzi $m0, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m0, $m5, $m0
	add $m6, $m6, $m0
	shl $m0, $m4, 2
	add $m6, $m6, $m0
	st32 $a7, $m6, $m15, 0
	f32sub $a5, $azero, $a7
	setzi $a6, 0
	setzi $a7, 0
	mov $m3, $m9
#if ATTENTION_FULL_BLOCK
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16 - 1
#else
	shr $m7, $m10, 4
	brz $m7, .Lsoftmax_exp_tail
	sub $m7, $m7, 1
#endif
	.macro SOFTMAX_EXP_PAIR
	f16v2tof32 $a0:1, $a0
	f32v2mul $a0:1, $a4:B, $a0:1
	f32v2add $a0:1, $a5:B, $a0:1
	f32v2tof16 $a0, $a0:1
	f16v2exp $a0, $a0
	{ st32step $a0, $mzero, $m2+=, 1
	  f16v2tof32 $a2:3, $a0 }
	.endm
.Lsoftmax_exp_panel:
	ld32step $a0, $mzero, $m3+=, 1
	.rept 7
	SOFTMAX_EXP_PAIR
	{ ld32step $a0, $mzero, $m3+=, 1
	  f32v2add $a6:7, $a6:7, $a2:3 }
	.endr
	SOFTMAX_EXP_PAIR
	f32v2add $a6:7, $a6:7, $a2:3
	add $m3, $m3, $m8
	add $m2, $m2, $m8
	brnzdec $m7, .Lsoftmax_exp_panel
#if !ATTENTION_FULL_BLOCK
.Lsoftmax_exp_tail:
	and $m7, $m10, 15
	shr $m7, $m7, 1
	brz $m7, .Lsoftmax_exp_odd
	sub $m7, $m7, 1
.Lsoftmax_exp_pair:
	ld32step $a0, $mzero, $m3+=, 1
	SOFTMAX_EXP_PAIR
	f32v2add $a6:7, $a6:7, $a2:3
	brnzdec $m7, .Lsoftmax_exp_pair
.Lsoftmax_exp_odd:
	and $m0, $m10, 1
	brz $m0, .Lsoftmax_zero_tail
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a0:1, $a4:B, $a0:1
	f32v2add $a0:1, $a5:B, $a0:1
	// Mask the unused lane before exponentiation; it contributes exactly zero.
	ldconst $a1, 0xc77fe000
	f32v2tof16 $a0, $a0:1
	f16v2exp $a0, $a0
	{ st32step $a0, $mzero, $m2+=, 1
	  f16v2tof32 $a2:3, $a0 }
	f32v2add $a6:7, $a6:7, $a2:3
.Lsoftmax_zero_tail:
	and $m7, $m10, 15
	add $m7, $m7, 1
	shr $m7, $m7, 1
	setzi $m0, 8
	sub $m7, $m0, $m7
	brz $m7, .Lsoftmax_zero_panels_setup
	sub $m7, $m7, 1
.Lsoftmax_zero_pair:
	st32step $mzero, $mzero, $m2+=, 1
	brnzdec $m7, .Lsoftmax_zero_pair
.Lsoftmax_zero_panels_setup:
	add $m2, $m2, $m8
	shr $m0, $m10, 4
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16 - 1
	sub $m7, $m7, $m0
	brz $m7, .Lsoftmax_sum_ready
	sub $m7, $m7, 1
.Lsoftmax_zero_panel:
	.rept 8
	st32step $mzero, $mzero, $m2+=, 1
	.endr
	add $m2, $m2, $m8
	brnzdec $m7, .Lsoftmax_zero_panel
.Lsoftmax_sum_ready:
#endif
	f32add $a6, $a6, $a7
	shl $m0, $m5, 2
	add $m6, $m6, $m0
	st32 $a6, $m6, $m15, 0
	add $m4, $m4, 6
	cmpult $m0, $m4, $m5
	brnz $m0, .Lsoftmax_row
.Lsoftmax_done:
	exitz $m15
