#include "arch/gc_tile_defines.h"

// Full key blocks, AMP-left panel order. Tail blocks retain the masked kernel.
// m2 weights, m3 scores, m4 query rows, m5 valid keys (equal to padded keys).
	.section .text.ATTENTION_SOFTMAX_SYMBOL,"ax",@progbits
	.globl ATTENTION_SOFTMAX_SYMBOL
	.p2align 2
	.type ATTENTION_SOFTMAX_SYMBOL,@function
ATTENTION_SOFTMAX_SYMBOL:
	.supervisor
	add $m11, $m11, -12
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	setzi $m0, .Lsoftmax_worker
	runall $m0, $m11, 0
	sync TEXCH_SYNCZONE_LOCAL
	add $m11, $m11, 12
	br $m10
	.size ATTENTION_SOFTMAX_SYMBOL, .-ATTENTION_SOFTMAX_SYMBOL

	.worker
	.p2align 3
.Lsoftmax_worker:
	ld32 $m5, $mvertex_base, $m15, 2
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
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16 - 1
.Lsoftmax_max_panel:
	ld32step $a0, $mzero, $m3+=, 1
	.rept 7
	{ ld32step $a0, $mzero, $m3+=, 1
	  f16v2max $a6, $a6, $a0 }
	.endr
	f16v2max $a6, $a6, $a0
	add $m3, $m3, $m8
	brnzdec $m7, .Lsoftmax_max_panel
	f16v2tof32 $a6:7, $a6
	f32max $a6, $a6, $a7
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
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16 - 1
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
	f32add $a6, $a6, $a7
	shl $m0, $m5, 2
	add $m6, $m6, $m0
	st32 $a6, $m6, $m15, 0
	add $m4, $m4, 6
	cmpult $m0, $m4, $m5
	brnz $m0, .Lsoftmax_row
.Lsoftmax_done:
	exitz $m15
