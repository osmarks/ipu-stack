#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set GELU_ALPHA_F16X2, 0x3a623a62
	.set GELU_BETA_F16X2, 0x29b929b9
	.set GELU_MIX_COEFFICIENTS_F16X2, 0x3a622891 // alpha*beta (low), alpha (high)
	.set ONE_F16X2, 0x3c003c00
	.set HALF_F16X2, 0x38003800
	// Callable supervisor ABI: m2 output, m3 input, m4 physical element
	// count, and m10 return. The contiguous entry accepts whole half pairs.

	.macro GELU_PAIR value
	// tanh GeLU: x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
	ld32 $a2, $mvertex_base, $m15, 10
	f16v2min $a3, \value, $a2
	ld32 $a2, $mvertex_base, $m15, 12
	f16v2max $a3, $a3, $a2
	f16v2mul $a2, $a3, $a3
	f16v2mul $a2, $a2, $a3
	f16v2mul $a2, $a4, $a2
	f16v2add $a2, $a3, $a2
	f16v2mul $a2, $a5, $a2
	f16v2tanh $a2, $a2
	f16v2add $a2, $a6, $a2
	f16v2mul $a2, $a7, $a2
	f16v2mul $a2, \value, $a2
	.endm

	// Two identical pairs per bound let the vector loop load both in one issue.
	.macro STORE_GELU_BOUNDS
	ldconst $m0, 0x48004800
	st32 $m0, $m11, $m15, 10
	st32 $m0, $m11, $m15, 11
	ldconst $m0, 0xc800c800
	st32 $m0, $m11, $m15, 12
	st32 $m0, $m11, $m15, 13
	.endm

	.macro LOAD_GELU_CONSTANTS
	ldconst $a4, GELU_BETA_F16X2
	ldconst $a5, GELU_ALPHA_F16X2
	ldconst $a6, ONE_F16X2
	ldconst $a7, HALF_F16X2
	// Bound the tanh polynomial, not the final x: |x|>=8 is saturated.
	// Bounds live in the worker arguments; only eight ARF registers are writable.
	.endm

	// Hardware repeat bodies require explicit bundles; ordinary loops retain
	// their compact single-pipeline encodings.
	.macro GELU_MAIN wide, instruction:vararg
	.if \wide
	{ \instruction
	  fnop }
	.else
	\instruction
	.endif
	.endm
	.macro GELU_AUX wide, instruction:vararg
	.if \wide
	{ nop
	  \instruction }
	.else
	\instruction
	.endif
	.endm

	.macro GELU_TWO_PAIRS offset, step=0
#ifdef GELU_WITH_BIAS
	.if \step
	GELU_MAIN \step, ld64step $a2:3, $mzero, $m9+=, 6
	.else
	GELU_MAIN \step, ld32 $a2, $m9, $m15, \offset
	GELU_MAIN \step, ld32 $a3, $m9, $m15, (\offset + 1)
	.endif
	// TAS holds the polynomial coefficients, freeing a4:5 for unclamped x+b.
	.if \step
	{ ld64 $a2:3, $mvertex_base, $m15, 5
	  f16v4add $a4:5, $a0:1, $a2:3 }
	{ ld64 $a2:3, $mvertex_base, $m15, 6
	  f16v4min $a0:1, $a4:5, $a2:3 }
	.else
	f16v4add $a4:5, $a0:1, $a2:3
	ld64 $a2:3, $mvertex_base, $m15, 5
	f16v4min $a0:1, $a4:5, $a2:3
	ld64 $a2:3, $mvertex_base, $m15, 6
	.endif
#else
	ld64 $a2:3, $mvertex_base, $m15, 5
	f16v4min $a0:1, $a0:1, $a2:3
	ld64 $a2:3, $mvertex_base, $m15, 6
#endif
	GELU_AUX \step, f16v4max $a0:1, $a0:1, $a2:3
	GELU_AUX \step, f16v4mul $a2:3, $a0:1, $a0:1
	GELU_AUX \step, f16v4mul $a2:3, $a2:3, $a0:1
#ifdef GELU_WITH_BIAS
	// alpha*beta*x^3 + alpha*x, with one FP32 dot product then FP16 rounding.
	GELU_AUX \step, f16v4mix $a0:1, $a2:3, $a0:1
	GELU_AUX \step, f16v4gacc $a2:3
#else
	GELU_AUX \step, f16v4mul $a2:3, $a4:BL, $a2:3
	GELU_AUX \step, f16v4add $a2:3, $a0:1, $a2:3
	GELU_AUX \step, f16v4mul $a2:3, $a5:BL, $a2:3
#endif
#ifdef GELU_WITH_BIAS
	GELU_AUX \step, f16v2tanh $a2, $a2
	GELU_AUX \step, f16v2tanh $a3, $a3
#else
	{ ld32 $a0, $m3, $m15, \offset
	  f16v2tanh $a2, $a2 }
	{ ld32 $a1, $m3, $m15, (\offset + 1)
	  f16v2tanh $a3, $a3 }
#endif
	GELU_AUX \step, f16v4add $a2:3, $a6:BL, $a2:3
	GELU_AUX \step, f16v4mul $a2:3, $a7:BL, $a2:3
#ifdef GELU_WITH_BIAS
	GELU_AUX \step, f16v4mul $a2:3, $a4:5, $a2:3
#else
	GELU_AUX \step, f16v4mul $a2:3, $a0:1, $a2:3
#endif
	.endm

	.macro GELU_EIGHT_PAIRS
	ld32 $a0, $m3, $m15, 0
	ld32 $a1, $m3, $m15, 1
	GELU_TWO_PAIRS 0
	st32 $a2, $m2, $m15, 0
	st32 $a3, $m2, $m15, 1
	ld32 $a0, $m3, $m15, 2
	ld32 $a1, $m3, $m15, 3
	GELU_TWO_PAIRS 2
	st32 $a2, $m2, $m15, 2
	st32 $a3, $m2, $m15, 3
	ld32 $a0, $m3, $m15, 4
	ld32 $a1, $m3, $m15, 5
	GELU_TWO_PAIRS 4
	st32 $a2, $m2, $m15, 4
	st32 $a3, $m2, $m15, 5
	ld32 $a0, $m3, $m15, 6
	ld32 $a1, $m3, $m15, 7
	GELU_TWO_PAIRS 6
	st32 $a2, $m2, $m15, 6
	st32 $a3, $m2, $m15, 7
	.endm


#ifndef GELU_MACROS_ONLY
	.macro GELU_ENTRY symbol, worker
	.section .text.\symbol,"ax",@progbits
	.globl \symbol
	.p2align 2
	.type \symbol,@function
\symbol:
	.supervisor
	add $m11, $m11, -64
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
#ifdef GELU_WITH_BIAS
	// m4 bias, m5 rows, m6 row width; the ordinary entry takes m4 elements.
	st32 $m5, $m11, $m15, 2
	st32 $m4, $m11, $m15, 5
	st32 $m6, $m11, $m15, 6
#else
	st32 $m4, $m11, $m15, 2
#endif
	STORE_GELU_BOUNDS
	setzi $m0, \worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 64
	br $m10
	.size \symbol, .-\symbol
	.endm

#ifdef GELU_WITH_BIAS
	GELU_ENTRY bias_gelu_f16, .Lgelu_contiguous_worker
#else
	GELU_ENTRY gelu_tanh_approx_f16, .Lgelu_contiguous_worker
#endif

	.worker
	.p2align 3
.Lgelu_contiguous_worker:
#ifdef GELU_WITH_BIAS
	ldconst $a0, GELU_MIX_COEFFICIENTS_F16X2
	uput $TAS, $a0
	setzi $a0, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
	uput $FP_CLR, $a0
	ld32 $m7, $mvertex_base, $m15, 2
	setzi $m1, 0
.Lgelu_row:
#endif
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
#ifdef GELU_WITH_BIAS
	ld32 $m8, $mvertex_base, $m15, 6
	ld32 $m9, $mvertex_base, $m15, 5
	add $m2, $m2, $m1
	add $m3, $m3, $m1
#else
	ld32 $m8, $mvertex_base, $m15, 2
#endif
	shr $m8, $m8, 1
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
#ifdef GELU_WITH_BIAS
	// Whole quads with naturally aligned pointers use a single hardware loop.
	// Other row widths and offset views retain the 32-bit path below.
	or $m0, $m2, $m3
	or $m0, $m0, $m9
	and $m0, $m0, 7
	and $m6, $m8, 1
	or $m0, $m0, $m6
	brnz $m0, .Lgelu_narrow
	shr $m6, $m8, 1
	cmpult $m0, $m4, $m6
	brz $m0, .Lgelu_contiguous_done
	add $m6, $m6, 5
	sub $m6, $m6, $m4
	setzi $m0, 43691 // ceil(quad_count / 6), for tile-sized arrays
	mul $m6, $m6, $m0
	shr $m6, $m6, 18
	shl $m0, $m4, 3
	add $m2, $m2, $m0
	add $m3, $m3, $m0
	add $m9, $m9, $m0
	ldconst $a6, ONE_F16X2
	ldconst $a7, HALF_F16X2
	.p2align 3
	// Fifteen issue groups per quad, including the dual-issued bound loads.
	{ rpt $m6, 14
	  fnop }
	{ ld64step $a0:1, $mzero, $m3+=, 6
	  fnop }
	GELU_TWO_PAIRS 0, 1
	{ st64step $a2:3, $mzero, $m2+=, 6
	  fnop }
	bri .Lgelu_contiguous_done
.Lgelu_narrow:
#endif
	shl $m4, $m4, 3
	cmpult $m6, $m4, $m8
	brz $m6, .Lgelu_contiguous_done
	LOAD_GELU_CONSTANTS
	shl $m5, $m4, 2
	add $m2, $m2, $m5
	add $m3, $m3, $m5
#ifdef GELU_WITH_BIAS
	add $m9, $m9, $m5
#endif
	setzi $m5, 8
	add $m6, $m5, -1
	and $m6, $m8, $m6
	brnz $m6, .Lgelu_contiguous_loop
.Lgelu_contiguous_aligned_loop:
	GELU_EIGHT_PAIRS
	add $m2, $m2, 192
	add $m3, $m3, 192
#ifdef GELU_WITH_BIAS
	add $m9, $m9, 192
#endif
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
#ifdef GELU_WITH_BIAS
	add $m9, $m9, 192
#endif
	add $m4, $m4, 48
	cmpult $m6, $m4, $m8
	brnz $m6, .Lgelu_contiguous_loop
	bri .Lgelu_contiguous_done
.Lgelu_contiguous_tail:
#ifdef GELU_WITH_BIAS
	// The vector path used a4:5 for x+b. Restore just these coefficients once;
	// the scalar loop preserves them, and a6:7 still hold one and one-half.
	ldconst $a4, GELU_BETA_F16X2
	ldconst $a5, GELU_ALPHA_F16X2
#endif
.Lgelu_pair_tail:
	ld32 $a0, $m3, $m15, 0
#ifdef GELU_WITH_BIAS
	ld32 $a2, $m9, $m15, 0
	f16v2add $a0, $a0, $a2
	add $m9, $m9, 4
#endif
	GELU_PAIR $a0
	st32 $a2, $m2, $m15, 0
	add $m2, $m2, 4
	add $m3, $m3, 4
	add $m4, $m4, 1
	cmpult $m6, $m4, $m8
	brnz $m6, .Lgelu_pair_tail
.Lgelu_contiguous_done:
#ifdef GELU_WITH_BIAS
	shl $m6, $m8, 2
	add $m1, $m1, $m6
	sub $m7, $m7, 1
	brnz $m7, .Lgelu_row
#endif
	exitz $m15

#endif
