#include "arch/gc_tile_defines.h"

#ifndef ATTENTION_HEAD_DIMENSION
#define ATTENTION_HEAD_DIMENSION 64
#endif
#ifndef ATTENTION_VALUE_DIMENSION
#define ATTENTION_VALUE_DIMENSION ATTENTION_HEAD_DIMENSION
#endif
#ifndef ATTENTION_PADDED_VALUE_DIMENSION
#define ATTENTION_PADDED_VALUE_DIMENSION ATTENTION_VALUE_DIMENSION
#endif
#ifndef ATTENTION_KEY_BLOCK_COLUMNS
#define ATTENTION_KEY_BLOCK_COLUMNS 64
#endif
#ifndef ATTENTION_SMALL_QUERY_ROWS
#define ATTENTION_SMALL_QUERY_ROWS 1
#endif
#ifndef ATTENTION_LARGE_QUERY_ROWS
#define ATTENTION_LARGE_QUERY_ROWS ATTENTION_SMALL_QUERY_ROWS
#endif
#ifndef ATTENTION_SMALL_KEY_ROWS
#define ATTENTION_SMALL_KEY_ROWS ATTENTION_KEY_BLOCK_COLUMNS
#endif
#ifndef ATTENTION_LARGE_KEY_ROWS
#define ATTENTION_LARGE_KEY_ROWS ATTENTION_KEY_BLOCK_COLUMNS
#endif
#ifndef ATTENTION_SCALE_BITS
#define ATTENTION_SCALE_BITS 0x3e000000
#endif

#define ATTENTION_WORKER_CONTEXTS 6
#define ATTENTION_PANEL_COLUMNS 16

#if ATTENTION_HEAD_DIMENSION <= 0
#error "attention head dimension must be positive"
#endif
#if ATTENTION_KEY_BLOCK_COLUMNS <= 0 || ATTENTION_KEY_BLOCK_COLUMNS % 16 != 0
#error "attention key block columns must be a positive multiple of 16"
#endif
#if ATTENTION_VALUE_DIMENSION <= 0 || ATTENTION_PADDED_VALUE_DIMENSION < ATTENTION_VALUE_DIMENSION
#error "attention value dimensions are invalid"
#endif

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set ATTENTION_MINIMUM_F32, 0xc77fe000
	.set ATTENTION_MINIMUM_F16X2, 0xfbfffbff
	.set ATTENTION_PHYSICAL_KEY_PAIRS, ATTENTION_KEY_BLOCK_COLUMNS / 2
	.set ATTENTION_VALUE_FULL_PANELS, ATTENTION_VALUE_DIMENSION / 16
	.set ATTENTION_VALUE_TAIL_PAIRS, (ATTENTION_VALUE_DIMENSION % 16 + 1) / 2

#if defined(ATTENTION_BUILD_ASSEMBLY_SOFTMAX_SMALL_KEY) || defined(ATTENTION_BUILD_ASSEMBLY_SOFTMAX_LARGE_KEY)
	// Callable supervisor ABI for softmax: m2 weights, m3 scores, m10 return.
	// Query/key row counts are specialization constants placed in the shared
	// worker argument block. The four public entries therefore share one worker
	// implementation rather than cloning its loops for every edge-block shape.
	.macro ATTENTION_SOFTMAX_ENTRY name, query_rows, key_rows
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	setzi $m0, \query_rows
	st32 $m0, $m11, $m15, 2
	setzi $m0, \key_rows
	st32 $m0, $m11, $m15, 3
#if ATTENTION_KEY_BLOCK_COLUMNS >= ATTENTION_WORKER_CONTEXTS * ATTENTION_PANEL_COLUMNS
	setzi $m0, .Lattention_softmax_panel_max_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	setzi $m0, .Lattention_softmax_reduce_max_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	setzi $m0, .Lattention_softmax_panel_exp_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	setzi $m0, .Lattention_softmax_reduce_sum_worker
	runall $m0, $m11, 0
#else
	setzi $m0, .Lattention_softmax_worker
	runall $m0, $m11, 0
#endif
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size \name, .-\name
	.endm

#ifdef ATTENTION_BUILD_ASSEMBLY_SOFTMAX_SMALL_KEY
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_small_query_small_key_f16, ATTENTION_SMALL_QUERY_ROWS, ATTENTION_SMALL_KEY_ROWS
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_large_query_small_key_f16, ATTENTION_LARGE_QUERY_ROWS, ATTENTION_SMALL_KEY_ROWS
#endif
#ifdef ATTENTION_BUILD_ASSEMBLY_SOFTMAX_LARGE_KEY
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_small_query_large_key_f16, ATTENTION_SMALL_QUERY_ROWS, ATTENTION_LARGE_KEY_ROWS
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_large_query_large_key_f16, ATTENTION_LARGE_QUERY_ROWS, ATTENTION_LARGE_KEY_ROWS
#endif

	.macro ATTENTION_SOFTMAX_MAX_PAIR
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32max $a7, $a7, $a0
	f32max $a7, $a7, $a1
	add $m3, $m3, 4
	.endm

	.macro ATTENTION_SOFTMAX_MAX_THREE_PAIRS
	ld32 $a0, $m3, $m15, 0
	ld32 $a2, $m3, $m15, 1
	ld32 $a4, $m3, $m15, 2
	f16v2tof32 $a0:1, $a0
	f16v2tof32 $a2:3, $a2
	f16v2tof32 $a4:5, $a4
	f32v2max $a0:1, $a0:1, $a2:3
	f32v2max $a0:1, $a0:1, $a4:5
	f32max $a0, $a0, $a1
	f32max $a7, $a7, $a0
	add $m3, $m3, 12
	.endm

	.macro ATTENTION_SOFTMAX_MAX_TWO_PAIRS
	ld32 $a0, $m3, $m15, 0
	ld32 $a2, $m3, $m15, 1
	f16v2tof32 $a0:1, $a0
	f16v2tof32 $a2:3, $a2
	f32v2max $a0:1, $a0:1, $a2:3
	f32max $a0, $a0, $a1
	f32max $a7, $a7, $a0
	add $m3, $m3, 8
	.endm

	.macro ATTENTION_SOFTMAX_MAX_PANEL
	ATTENTION_SOFTMAX_MAX_THREE_PAIRS
	ATTENTION_SOFTMAX_MAX_THREE_PAIRS
	ATTENTION_SOFTMAX_MAX_TWO_PAIRS
	.endm

	.macro ATTENTION_SOFTMAX_NORMALIZE_PAIR
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a2:3, $a4:B, $a0:1
	f32v2add $a2:3, $a5:B, $a2:3
	f32v2tof16 $a2, $a2:3
	st32 $a2, $m2, $m15, 0
	add $m2, $m2, 4
	add $m3, $m3, 4
	.endm

	.macro ATTENTION_SOFTMAX_NORMALIZE_TWO_PAIRS
	ld32 $a0, $m3, $m15, 0
	ld32 $a2, $m3, $m15, 1
	f16v2tof32 $a0:1, $a0
	f16v2tof32 $a2:3, $a2
	f32v2mul $a0:1, $a4:B, $a0:1
	f32v2mul $a2:3, $a4:B, $a2:3
	f32v2add $a0:1, $a5:B, $a0:1
	f32v2add $a2:3, $a5:B, $a2:3
	f32v2tof16 $a0, $a0:1
	f32v2tof16 $a2, $a2:3
	st32 $a0, $m2, $m15, 0
	st32 $a2, $m2, $m15, 1
	add $m2, $m2, 8
	add $m3, $m3, 8
	.endm

	.macro ATTENTION_SOFTMAX_SUM_PAIR
	ld32 $a0, $m2, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32add $a7, $a7, $a0
	f32add $a7, $a7, $a1
	add $m2, $m2, 4
	.endm

	.macro ATTENTION_SOFTMAX_SUM_THREE_PAIRS
	ld32 $a0, $m2, $m15, 0
	ld32 $a2, $m2, $m15, 1
	ld32 $a4, $m2, $m15, 2
	f16v2tof32 $a0:1, $a0
	f16v2tof32 $a2:3, $a2
	f16v2tof32 $a4:5, $a4
	f32v2add $a0:1, $a0:1, $a2:3
	f32v2add $a0:1, $a0:1, $a4:5
	f32add $a0, $a0, $a1
	f32add $a7, $a7, $a0
	add $m2, $m2, 12
	.endm

	.macro ATTENTION_SOFTMAX_SUM_TWO_PAIRS
	ld32 $a0, $m2, $m15, 0
	ld32 $a2, $m2, $m15, 1
	f16v2tof32 $a0:1, $a0
	f16v2tof32 $a2:3, $a2
	f32v2add $a0:1, $a0:1, $a2:3
	f32add $a0, $a0, $a1
	f32add $a7, $a7, $a0
	add $m2, $m2, 8
	.endm

	.macro ATTENTION_SOFTMAX_SUM_PANEL
	ATTENTION_SOFTMAX_SUM_THREE_PAIRS
	ATTENTION_SOFTMAX_SUM_THREE_PAIRS
	ATTENTION_SOFTMAX_SUM_TWO_PAIRS
	.endm

	.worker
	.p2align 3
#if ATTENTION_KEY_BLOCK_COLUMNS < ATTENTION_WORKER_CONTEXTS * ATTENTION_PANEL_COLUMNS
.Lattention_softmax_worker:
	ld32 $m5, $mvertex_base, $m15, 2
	ld32 $m6, $mvertex_base, $m15, 3
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m7, $m4, $m5
	brz $m7, .Lattention_softmax_done
	// Consecutive 16-column panels are separated by the other query rows.
	add $m8, $m5, -1
	shl $m8, $m8, 5
.Lattention_softmax_row:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	shl $m10, $m4, 5
	add $m2, $m2, $m10
	add $m3, $m3, $m10

	// Scores originate in F16, so reducing their maximum in F32 is exact while
	// avoiding scalar address reconstruction.
	ldconst $a7, ATTENTION_MINIMUM_F32
	shr $m7, $m6, 4
	brz $m7, .Lattention_softmax_max_tail_setup
.Lattention_softmax_max_panels:
	ATTENTION_SOFTMAX_MAX_PANEL
	add $m3, $m3, $m8
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_max_panels
.Lattention_softmax_max_tail_setup:
	and $m7, $m6, 15
	shr $m7, $m7, 1
	brz $m7, .Lattention_softmax_max_odd
.Lattention_softmax_max_tail:
	ATTENTION_SOFTMAX_MAX_PAIR
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_max_tail
.Lattention_softmax_max_odd:
	and $m10, $m6, 1
	brz $m10, .Lattention_softmax_max_ready
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32max $a7, $a7, $a0
.Lattention_softmax_max_ready:
	ldconst $a4, ATTENTION_SCALE_BITS
	f32mul $a7, $a7, $a4
	f32sub $a5, $azero, $a7

	// First round normalized scores to F16. Exponentiation is kept in a separate
	// software pipeline below: f16v2exp has enough latency that consuming every
	// result immediately would serialize the entire row.
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	shl $m10, $m4, 5
	add $m2, $m2, $m10
	add $m3, $m3, $m10
	shr $m7, $m6, 4
	brz $m7, .Lattention_softmax_norm_tail_setup
.Lattention_softmax_norm_panels:
	.rept 4
	ATTENTION_SOFTMAX_NORMALIZE_TWO_PAIRS
	.endr
	add $m2, $m2, $m8
	add $m3, $m3, $m8
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_norm_panels
.Lattention_softmax_norm_tail_setup:
	and $m7, $m6, 15
	shr $m7, $m7, 1
	brz $m7, .Lattention_softmax_norm_odd
.Lattention_softmax_norm_tail:
	ATTENTION_SOFTMAX_NORMALIZE_PAIR
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_norm_tail
.Lattention_softmax_norm_odd:
	and $m10, $m6, 1
	brz $m10, .Lattention_softmax_padding_setup
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a2:3, $a4:B, $a0:1
	f32v2add $a2:3, $a5:B, $a2:3
	// The high lane is outside the logical key block. Rounding -65504 before the
	// exponentiation pass guarantees that the stored padding lane becomes zero.
	ldconst $a3, ATTENTION_MINIMUM_F32
	f32v2tof16 $a2, $a2:3
	st32 $a2, $m2, $m15, 0
	add $m2, $m2, 4
.Lattention_softmax_padding_setup:
	add $m7, $m6, 1
	shr $m7, $m7, 1
	mov $m9, $m7
	setzi $m10, ATTENTION_PHYSICAL_KEY_PAIRS
	sub $m7, $m10, $m7
	brz $m7, .Lattention_softmax_exp_setup
	ldconst $a0, ATTENTION_MINIMUM_F16X2
.Lattention_softmax_zero_pairs:
	st32 $a0, $m2, $m15, 0
	add $m2, $m2, 4
	add $m9, $m9, 1
	and $m10, $m9, 7
	brnz $m10, .Lattention_softmax_zero_no_panel_step
	add $m2, $m2, $m8
.Lattention_softmax_zero_no_panel_step:
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_zero_pairs

.Lattention_softmax_exp_setup:
	// The maximum is complete before the exp/sum passes, so commit it now and
	// reuse its ARF register for a tree-accumulated denominator.
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m10, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m10, $m5, $m10
	add $m2, $m2, $m10
	shl $m10, $m4, 2
	add $m2, $m2, $m10
	st32 $a7, $m2, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 0
	shl $m10, $m4, 5
	add $m1, $m1, $m10
	add $m10, $m8, 32
	setzi $m0, ATTENTION_KEY_BLOCK_COLUMNS / 16
.Lattention_softmax_exp_panel:
	mov $m2, $m1
	mov $m3, $m1
	setzi $m7, 4
	// Four independent half4 groups keep the nonlinear pipeline occupied. This
	// is the same schedule used by the standalone vector exp implementation.
	brnzdec $m7, 3f
	bri 4f
	.p2align 3
	nop
3:
	ld64step $a0:1, $mzero, $m2+=, 1
	{ rpt $m7, (2f-1f)/8 - 1
	  f16v2exp $a2, $a0 }
1:
	{ ld64step $a0:1, $mzero, $m2+=, 1
	  f16v2exp $a3, $a1 }
	{ st64step $a2:3, $mzero, $m3+=, 1
	  f16v2exp $a2, $a0 }
2:
	f16v2exp $a3, $a1
	st64step $a2:3, $mzero, $m3+=, 1
4:
	add $m1, $m1, $m10
	sub $m0, $m0, 1
	brnz $m0, .Lattention_softmax_exp_panel

	// Accumulate after the nonlinear pipeline so no exp result feeds directly
	// into the loop-carried denominator dependency.
	ld32 $m2, $mvertex_base, $m15, 0
	shl $m10, $m4, 5
	add $m2, $m2, $m10
	shr $m7, $m6, 4
	ldconst $a7, 0
	brz $m7, .Lattention_softmax_sum_tail_setup
.Lattention_softmax_sum_panels:
	ATTENTION_SOFTMAX_SUM_PANEL
	add $m2, $m2, $m8
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_sum_panels
.Lattention_softmax_sum_tail_setup:
	and $m7, $m6, 15
	shr $m7, $m7, 1
	brz $m7, .Lattention_softmax_sum_odd
.Lattention_softmax_sum_tail:
	ATTENTION_SOFTMAX_SUM_PAIR
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_sum_tail
.Lattention_softmax_sum_odd:
	and $m10, $m6, 1
	brz $m10, .Lattention_softmax_store_state
	ld32 $a0, $m2, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32add $a7, $a7, $a0
.Lattention_softmax_store_state:
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m10, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m10, $m5, $m10
	add $m2, $m2, $m10
	shl $m10, $m5, 2
	add $m2, $m2, $m10
	shl $m10, $m4, 2
	add $m2, $m2, $m10
	st32 $a7, $m2, $m15, 0
	add $m4, $m4, ATTENTION_WORKER_CONTEXTS
	cmpult $m7, $m4, $m5
	brnz $m7, .Lattention_softmax_row
.Lattention_softmax_done:
	exitz $m15
#else
	// For wide rows, assign 16-column panels rather than complete rows to
	// workers. Six partial maxima and sums are reduced by short worker waves;
	// the extra state panel already reserved after the probability matrix holds
	// both sets of partials without increasing the allocation.
.Lattention_softmax_panel_max_worker:
	ld32 $m5, $mvertex_base, $m15, 2
	ld32 $m6, $mvertex_base, $m15, 3
	get $m1, $WSR
	and $m1, $m1, CSR_W_WSR__CTXTID_M1__MASK
	shl $m8, $m5, 5
	setzi $m0, 0
.Lattention_softmax_panel_max_row:
	ldconst $a7, ATTENTION_MINIMUM_F32
	mov $m4, $m1
.Lattention_softmax_panel_max_panel:
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16
	cmpult $m7, $m4, $m7
	brz $m7, .Lattention_softmax_panel_max_store
	shl $m10, $m4, 4
	cmpult $m7, $m10, $m6
	brz $m7, .Lattention_softmax_panel_max_next
	mul $m7, $m4, $m8
	ld32 $m3, $mvertex_base, $m15, 1
	add $m3, $m3, $m7
	shl $m7, $m0, 5
	add $m3, $m3, $m7
	sub $m7, $m6, $m10
	setzi $m9, 16
	cmpult $m9, $m7, $m9
	brnz $m9, .Lattention_softmax_panel_max_partial
	ATTENTION_SOFTMAX_MAX_PANEL
	bri .Lattention_softmax_panel_max_next
.Lattention_softmax_panel_max_partial:
	shr $m9, $m7, 1
	brz $m9, .Lattention_softmax_panel_max_partial_odd
.Lattention_softmax_panel_max_partial_pairs:
	ATTENTION_SOFTMAX_MAX_PAIR
	sub $m9, $m9, 1
	brnz $m9, .Lattention_softmax_panel_max_partial_pairs
.Lattention_softmax_panel_max_partial_odd:
	and $m10, $m7, 1
	brz $m10, .Lattention_softmax_panel_max_next
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32max $a7, $a7, $a0
.Lattention_softmax_panel_max_next:
	add $m4, $m4, 6
	bri .Lattention_softmax_panel_max_panel
.Lattention_softmax_panel_max_store:
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m7, $m5, $m7
	add $m2, $m2, $m7
	mul $m7, $m1, $m5
	add $m7, $m7, $m0
	shl $m7, $m7, 2
	add $m2, $m2, $m7
	st32 $a7, $m2, $m15, 0
	add $m0, $m0, 1
	cmpult $m7, $m0, $m5
	brnz $m7, .Lattention_softmax_panel_max_row
	exitz $m15

.Lattention_softmax_reduce_max_worker:
	ld32 $m5, $mvertex_base, $m15, 2
	get $m0, $WSR
	and $m0, $m0, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m7, $m0, $m5
	brz $m7, .Lattention_softmax_reduce_max_done
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m7, $m5, $m7
	add $m2, $m2, $m7
	shl $m8, $m5, 2
.Lattention_softmax_reduce_max_row:
	shl $m7, $m0, 2
	add $m9, $m2, $m7
	mov $m3, $m9
	ld32 $a7, $m3, $m15, 0
	.rept ATTENTION_WORKER_CONTEXTS - 1
	add $m3, $m3, $m8
	ld32 $a0, $m3, $m15, 0
	f32max $a7, $a7, $a0
	.endr
	ldconst $a4, ATTENTION_SCALE_BITS
	f32mul $a7, $a7, $a4
	st32 $a7, $m9, $m15, 0
	add $m0, $m0, ATTENTION_WORKER_CONTEXTS
	cmpult $m7, $m0, $m5
	brnz $m7, .Lattention_softmax_reduce_max_row
.Lattention_softmax_reduce_max_done:
	exitz $m15

.Lattention_softmax_panel_exp_worker:
	ld32 $m5, $mvertex_base, $m15, 2
	ld32 $m6, $mvertex_base, $m15, 3
	get $m1, $WSR
	and $m1, $m1, CSR_W_WSR__CTXTID_M1__MASK
	shl $m8, $m5, 5
	ld32 $m9, $mvertex_base, $m15, 0
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m7, $m5, $m7
	add $m9, $m9, $m7
	ldconst $a4, ATTENTION_SCALE_BITS
	mov $m4, $m1
.Lattention_softmax_panel_normalize_panel:
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / 16
	cmpult $m7, $m4, $m7
	brz $m7, .Lattention_softmax_panel_sum_setup
	mul $m7, $m4, $m8
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	add $m2, $m2, $m7
	add $m3, $m3, $m7
	shl $m10, $m4, 4
	cmpult $m7, $m10, $m6
	brz $m7, .Lattention_softmax_panel_zero
	sub $m10, $m6, $m10
	setzi $m7, 16
	cmpult $m7, $m10, $m7
	brnz $m7, .Lattention_softmax_panel_normalize_partial
	mov $m0, $m5
	mov $m7, $m9
.Lattention_softmax_panel_normalize_full_row:
	ld32 $a7, $m7, $m15, 0
	add $m7, $m7, 4
	f32sub $a5, $azero, $a7
	.rept 4
	ATTENTION_SOFTMAX_NORMALIZE_TWO_PAIRS
	.endr
	sub $m0, $m0, 1
	brnz $m0, .Lattention_softmax_panel_normalize_full_row
	bri .Lattention_softmax_panel_exp
.Lattention_softmax_panel_normalize_partial:
	mov $m0, $m5
	mov $m7, $m9
.Lattention_softmax_panel_normalize_partial_row:
	ld32 $a7, $m7, $m15, 0
	add $m7, $m7, 4
	f32sub $a5, $azero, $a7
	shr $m6, $m10, 1
	brz $m6, .Lattention_softmax_panel_normalize_partial_odd
.Lattention_softmax_panel_normalize_partial_pairs:
	ATTENTION_SOFTMAX_NORMALIZE_PAIR
	sub $m6, $m6, 1
	brnz $m6, .Lattention_softmax_panel_normalize_partial_pairs
.Lattention_softmax_panel_normalize_partial_odd:
	and $m6, $m10, 1
	brz $m6, .Lattention_softmax_panel_normalize_partial_fill
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a2:3, $a4:B, $a0:1
	f32v2add $a2:3, $a5:B, $a2:3
	ldconst $a3, ATTENTION_MINIMUM_F32
	f32v2tof16 $a2, $a2:3
	st32 $a2, $m2, $m15, 0
	add $m2, $m2, 4
	add $m3, $m3, 4
.Lattention_softmax_panel_normalize_partial_fill:
	ldconst $a0, ATTENTION_MINIMUM_F16X2
	add $m6, $m10, 1
	shr $m6, $m6, 1
	setzi $m3, 8
	sub $m6, $m3, $m6
	brz $m6, .Lattention_softmax_panel_normalize_partial_next
.Lattention_softmax_panel_normalize_partial_fill_pairs:
	st32 $a0, $m2, $m15, 0
	add $m2, $m2, 4
	sub $m6, $m6, 1
	brnz $m6, .Lattention_softmax_panel_normalize_partial_fill_pairs
.Lattention_softmax_panel_normalize_partial_next:
	sub $m0, $m0, 1
	brz $m0, .Lattention_softmax_panel_exp
	// Reload the next source row because m3 was used as a fill-loop temporary.
	ld32 $m3, $mvertex_base, $m15, 1
	mul $m6, $m4, $m8
	add $m3, $m3, $m6
	sub $m6, $m5, $m0
	shl $m6, $m6, 5
	add $m3, $m3, $m6
	bri .Lattention_softmax_panel_normalize_partial_row
.Lattention_softmax_panel_zero:
	shl $m7, $m5, 2
.Lattention_softmax_panel_zero_loop:
	st64step $azeros, $mzero, $m2+=, 1
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_panel_zero_loop
	bri .Lattention_softmax_panel_normalize_next
.Lattention_softmax_panel_exp:
	ld32 $m2, $mvertex_base, $m15, 0
	mul $m7, $m4, $m8
	add $m2, $m2, $m7
	mov $m3, $m2
	shl $m7, $m5, 2
	brnzdec $m7, 3f
	bri 4f
	.p2align 3
	nop
3:
	ld64step $a0:1, $mzero, $m2+=, 1
	{ rpt $m7, (2f-1f)/8 - 1
	  f16v2exp $a2, $a0 }
1:
	{ ld64step $a0:1, $mzero, $m2+=, 1
	  f16v2exp $a3, $a1 }
	{ st64step $a2:3, $mzero, $m3+=, 1
	  f16v2exp $a2, $a0 }
2:
	f16v2exp $a3, $a1
	st64step $a2:3, $mzero, $m3+=, 1
4:
.Lattention_softmax_panel_normalize_next:
	ld32 $m6, $mvertex_base, $m15, 3
	add $m4, $m4, ATTENTION_WORKER_CONTEXTS
	bri .Lattention_softmax_panel_normalize_panel

.Lattention_softmax_panel_sum_setup:
	setzi $m0, 0
.Lattention_softmax_panel_sum_row:
	ldconst $a7, 0
	mov $m4, $m1
.Lattention_softmax_panel_sum_panel:
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS / ATTENTION_PANEL_COLUMNS
	cmpult $m7, $m4, $m7
	brz $m7, .Lattention_softmax_panel_sum_store
	mul $m7, $m4, $m8
	ld32 $m2, $mvertex_base, $m15, 0
	add $m2, $m2, $m7
	shl $m7, $m0, 5
	add $m2, $m2, $m7
	ld64step $a0:1, $mzero, $m2+=, 1
	ld64step $a2:3, $mzero, $m2+=, 1
	f16v2add $a0, $a0, $a1
	f16v2add $a2, $a2, $a3
	f16v2add $a0, $a0, $a2
	f16v2tof32 $a0:1, $a0
	f32add $a0, $a0, $a1
	f32add $a7, $a7, $a0
	ld64step $a0:1, $mzero, $m2+=, 1
	ld64step $a2:3, $mzero, $m2+=, 1
	f16v2add $a0, $a0, $a1
	f16v2add $a2, $a2, $a3
	f16v2add $a0, $a0, $a2
	f16v2tof32 $a0:1, $a0
	f32add $a0, $a0, $a1
	f32add $a7, $a7, $a0
	add $m4, $m4, ATTENTION_WORKER_CONTEXTS
	bri .Lattention_softmax_panel_sum_panel
.Lattention_softmax_panel_sum_store:
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m7, $m5, $m7
	add $m2, $m2, $m7
	shl $m7, $m5, 2
	add $m2, $m2, $m7
	mul $m7, $m1, $m5
	add $m7, $m7, $m0
	shl $m7, $m7, 2
	add $m2, $m2, $m7
	st32 $a7, $m2, $m15, 0
	add $m0, $m0, 1
	cmpult $m7, $m0, $m5
	brnz $m7, .Lattention_softmax_panel_sum_row
	exitz $m15

.Lattention_softmax_reduce_sum_worker:
	ld32 $m5, $mvertex_base, $m15, 2
	get $m0, $WSR
	and $m0, $m0, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m7, $m0, $m5
	brz $m7, .Lattention_softmax_reduce_sum_done
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m7, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m7, $m5, $m7
	add $m2, $m2, $m7
	shl $m8, $m5, 2
	add $m2, $m2, $m8
.Lattention_softmax_reduce_sum_row:
	shl $m7, $m0, 2
	add $m9, $m2, $m7
	mov $m3, $m9
	ld32 $a7, $m3, $m15, 0
	.rept ATTENTION_WORKER_CONTEXTS - 1
	add $m3, $m3, $m8
	ld32 $a0, $m3, $m15, 0
	f32add $a7, $a7, $a0
	.endr
	st32 $a7, $m9, $m15, 0
	add $m0, $m0, ATTENTION_WORKER_CONTEXTS
	cmpult $m7, $m0, $m5
	brnz $m7, .Lattention_softmax_reduce_sum_row
.Lattention_softmax_reduce_sum_done:
	exitz $m15
#endif
#endif

	// Callable supervisor ABI for merge: m2 accumulator, m3 block values,
	// m4 softmax state, m5 initial-block flag, m6 final-block flag, m10 return.
	.macro ATTENTION_MERGE_ENTRY name, query_rows
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -24
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	setzi $m0, \query_rows
	st32 $m0, $m11, $m15, 5
	setzi $m0, .Lattention_merge_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size \name, .-\name
	.endm

	ATTENTION_MERGE_ENTRY ipu_stack_attention_merge_small_query_f16, ATTENTION_SMALL_QUERY_ROWS
	ATTENTION_MERGE_ENTRY ipu_stack_attention_merge_large_query_f16, ATTENTION_LARGE_QUERY_ROWS

	.macro ATTENTION_MERGE_INITIAL_PAIR
	ld32 $a0, $m1, $m15, 0
	f16v2tof32 $a0:1, $a0
	st64 $a0:1, $m0, $m15, 0
	add $m1, $m1, 4
	add $m0, $m0, 8
	.endm

	.macro ATTENTION_MERGE_UPDATE_PAIR
	ld32 $a0, $m1, $m15, 0
	f16v2tof32 $a0:1, $a0
	ld64 $a2:3, $m0, $m15, 0
	f32v2mul $a2:3, $a5:B, $a2:3
	f32v2mul $a0:1, $a6:B, $a0:1
	f32v2add $a2:3, $a2:3, $a0:1
	st64 $a2:3, $m0, $m15, 0
	add $m1, $m1, 4
	add $m0, $m0, 8
	.endm

	.macro ATTENTION_MERGE_NORMALIZE_PAIR
	ld64 $a0:1, $m0, $m15, 0
	f32v2mul $a0:1, $a7:B, $a0:1
	st64 $a0:1, $m0, $m15, 0
	add $m0, $m0, 8
	.endm

	.worker
	.p2align 3
.Lattention_merge_worker:
	ld32 $m5, $mvertex_base, $m15, 5
	ld32 $m6, $mvertex_base, $m15, 3
	ld32 $m7, $mvertex_base, $m15, 4
	get $m9, $WSR
	and $m9, $m9, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m0, $m9, $m5
	brz $m0, .Lattention_merge_done
	add $m10, $m5, -1
	shl $m10, $m10, 5
.Lattention_merge_row:
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	shl $m8, $m9, 5
	add $m1, $m1, $m8
	setzi $m8, ATTENTION_PADDED_VALUE_DIMENSION * 4
	mul $m8, $m9, $m8
	add $m0, $m0, $m8
	// m8 remains the address of the two persistent online-softmax scalars.
	add $m8, $m0, ATTENTION_VALUE_DIMENSION * 4
	ld32 $m4, $mvertex_base, $m15, 2
	setzi $m2, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m2, $m5, $m2
	add $m4, $m4, $m2
	shl $m2, $m9, 2
	add $m4, $m4, $m2
	ld32 $a0, $m4, $m15, 0
	shl $m2, $m5, 2
	add $m4, $m4, $m2
	ld32 $a1, $m4, $m15, 0
	brz $m6, .Lattention_merge_update_state
	st32 $a0, $m8, $m15, 0
	st32 $a1, $m8, $m15, 1
	.if ATTENTION_VALUE_FULL_PANELS
	setzi $m2, ATTENTION_VALUE_FULL_PANELS
.Lattention_merge_initial_panels:
	.rept 8
	ATTENTION_MERGE_INITIAL_PAIR
	.endr
	add $m1, $m1, $m10
	sub $m2, $m2, 1
	brnz $m2, .Lattention_merge_initial_panels
	.endif
	.rept ATTENTION_VALUE_TAIL_PAIRS
	ATTENTION_MERGE_INITIAL_PAIR
	.endr
	bri .Lattention_merge_maybe_normalize

.Lattention_merge_update_state:
	ld32 $a2, $m8, $m15, 0
	ld32 $a3, $m8, $m15, 1
	f32max $a4, $a2, $a0
	f32sub $a5, $a2, $a4
	f32exp $a5, $a5
	f32sub $a6, $a0, $a4
	f32exp $a6, $a6
	f32mul $a3, $a3, $a5
	f32mul $a1, $a1, $a6
	f32add $a3, $a3, $a1
	st32 $a4, $m8, $m15, 0
	st32 $a3, $m8, $m15, 1
	.if ATTENTION_VALUE_FULL_PANELS
	setzi $m2, ATTENTION_VALUE_FULL_PANELS
.Lattention_merge_update_panels:
	.rept 8
	ATTENTION_MERGE_UPDATE_PAIR
	.endr
	add $m1, $m1, $m10
	sub $m2, $m2, 1
	brnz $m2, .Lattention_merge_update_panels
	.endif
	.rept ATTENTION_VALUE_TAIL_PAIRS
	ATTENTION_MERGE_UPDATE_PAIR
	.endr

.Lattention_merge_maybe_normalize:
	brz $m7, .Lattention_merge_next_row
	ld32 $a7, $m8, $m15, 1
	f32oox $a7, $a7
	ld32 $m0, $mvertex_base, $m15, 0
	setzi $m2, ATTENTION_PADDED_VALUE_DIMENSION * 4
	mul $m2, $m9, $m2
	add $m0, $m0, $m2
	setzi $m2, ATTENTION_VALUE_FULL_PANELS
	.if ATTENTION_VALUE_FULL_PANELS
.Lattention_merge_normalize_panels:
	.rept 8
	ATTENTION_MERGE_NORMALIZE_PAIR
	.endr
	sub $m2, $m2, 1
	brnz $m2, .Lattention_merge_normalize_panels
	.endif
	.rept ATTENTION_VALUE_TAIL_PAIRS
	ATTENTION_MERGE_NORMALIZE_PAIR
	.endr
.Lattention_merge_next_row:
	add $m9, $m9, 6
	cmpult $m0, $m9, $m5
	brnz $m0, .Lattention_merge_row
.Lattention_merge_done:
	exitz $m15
