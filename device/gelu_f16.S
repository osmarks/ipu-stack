#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set GELU_ALPHA_F16X2, 0x3a623a62
	.set GELU_BETA_F16X2, 0x29b929b9
	.set ONE_F16X2, 0x3c003c00
	.set HALF_F16X2, 0x38003800
	// Callable supervisor ABI: m2 output, m3 input, m4 physical element
	// count, and m10 return. The contiguous entry accepts whole half pairs.

	.macro GELU_PAIR value
	// tanh GeLU: x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
	f16v2mul $a2, \value, \value
	f16v2mul $a2, $a2, \value
	f16v2mul $a2, $a4, $a2
	f16v2add $a2, \value, $a2
	f16v2mul $a2, $a5, $a2
	f16v2tanh $a2, $a2
	f16v2add $a2, $a6, $a2
	f16v2mul $a2, $a7, $a2
	f16v2mul $a2, \value, $a2
	.endm

	.macro LOAD_GELU_CONSTANTS
	ldconst $a4, GELU_BETA_F16X2
	ldconst $a5, GELU_ALPHA_F16X2
	ldconst $a6, ONE_F16X2
	ldconst $a7, HALF_F16X2
	.endm

	.macro GELU_TWO_PAIRS
	f16v2mul $a2, $a0, $a0
	f16v2mul $a3, $a1, $a1
	f16v2mul $a2, $a2, $a0
	f16v2mul $a3, $a3, $a1
	f16v2mul $a2, $a4, $a2
	f16v2mul $a3, $a4, $a3
	f16v2add $a2, $a0, $a2
	f16v2add $a3, $a1, $a3
	f16v2mul $a2, $a5, $a2
	f16v2mul $a3, $a5, $a3
	f16v2tanh $a2, $a2
	f16v2tanh $a3, $a3
	f16v2add $a2, $a6, $a2
	f16v2add $a3, $a6, $a3
	f16v2mul $a2, $a7, $a2
	f16v2mul $a3, $a7, $a3
	f16v2mul $a2, $a0, $a2
	f16v2mul $a3, $a1, $a3
	.endm

	.macro GELU_EIGHT_PAIRS
	ld32 $a0, $m3, $m15, 0
	ld32 $a1, $m3, $m15, 1
	GELU_TWO_PAIRS
	st32 $a2, $m2, $m15, 0
	st32 $a3, $m2, $m15, 1
	ld32 $a0, $m3, $m15, 2
	ld32 $a1, $m3, $m15, 3
	GELU_TWO_PAIRS
	st32 $a2, $m2, $m15, 2
	st32 $a3, $m2, $m15, 3
	ld32 $a0, $m3, $m15, 4
	ld32 $a1, $m3, $m15, 5
	GELU_TWO_PAIRS
	st32 $a2, $m2, $m15, 4
	st32 $a3, $m2, $m15, 5
	ld32 $a0, $m3, $m15, 6
	ld32 $a1, $m3, $m15, 7
	GELU_TWO_PAIRS
	st32 $a2, $m2, $m15, 6
	st32 $a3, $m2, $m15, 7
	.endm

	.macro GELU_ENTRY symbol, worker
	.section .text.\symbol,"ax",@progbits
	.globl \symbol
	.p2align 2
	.type \symbol,@function
\symbol:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	setzi $m0, \worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size \symbol, .-\symbol
	.endm

	GELU_ENTRY ipu_stack_gelu_exact_f16, .Lgelu_contiguous_worker

	.worker
	.p2align 3
.Lgelu_contiguous_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m8, $mvertex_base, $m15, 2
	shr $m8, $m8, 1
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	shl $m4, $m4, 3
	cmpult $m6, $m4, $m8
	brz $m6, .Lgelu_contiguous_done
	LOAD_GELU_CONSTANTS
	shl $m5, $m4, 2
	add $m2, $m2, $m5
	add $m3, $m3, $m5
	setzi $m5, 8
	add $m6, $m5, -1
	and $m6, $m8, $m6
	brnz $m6, .Lgelu_contiguous_loop
.Lgelu_contiguous_aligned_loop:
	GELU_EIGHT_PAIRS
	add $m2, $m2, 192
	add $m3, $m3, 192
	add $m4, $m4, 48
	cmpult $m6, $m4, $m8
	brnz $m6, .Lgelu_contiguous_aligned_loop
	bri .Lgelu_contiguous_done
.Lgelu_contiguous_loop:
	sub $m6, $m8, $m4
	cmpult $m6, $m6, $m5
	brnz $m6, .Lgelu_contiguous_tail
	GELU_EIGHT_PAIRS
	add $m2, $m2, 192
	add $m3, $m3, 192
	add $m4, $m4, 48
	cmpult $m6, $m4, $m8
	brnz $m6, .Lgelu_contiguous_loop
	bri .Lgelu_contiguous_done
.Lgelu_contiguous_tail:
	ld32 $a0, $m3, $m15, 0
	GELU_PAIR $a0
	st32 $a2, $m2, $m15, 0
	add $m2, $m2, 4
	add $m3, $m3, 4
	add $m4, $m4, 1
	cmpult $m6, $m4, $m8
	brnz $m6, .Lgelu_contiguous_tail
.Lgelu_contiguous_done:
	exitz $m15
