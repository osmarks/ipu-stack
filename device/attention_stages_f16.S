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
	.set ATTENTION_PHYSICAL_KEY_PAIRS, ATTENTION_KEY_BLOCK_COLUMNS / 2
	.set ATTENTION_VALUE_PAIRS, (ATTENTION_VALUE_DIMENSION + 1) / 2

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
	setzi $m0, .Lattention_softmax_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size \name, .-\name
	.endm

	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_small_query_small_key_f16, ATTENTION_SMALL_QUERY_ROWS, ATTENTION_SMALL_KEY_ROWS
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_small_query_large_key_f16, ATTENTION_SMALL_QUERY_ROWS, ATTENTION_LARGE_KEY_ROWS
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_large_query_small_key_f16, ATTENTION_LARGE_QUERY_ROWS, ATTENTION_SMALL_KEY_ROWS
	ATTENTION_SOFTMAX_ENTRY ipu_stack_attention_softmax_large_query_large_key_f16, ATTENTION_LARGE_QUERY_ROWS, ATTENTION_LARGE_KEY_ROWS

	.worker
	.p2align 3
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
	shr $m7, $m6, 1
	setzi $m9, 0
	brz $m7, .Lattention_softmax_max_odd
.Lattention_softmax_max_pairs:
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32max $a7, $a7, $a0
	f32max $a7, $a7, $a1
	add $m3, $m3, 4
	add $m9, $m9, 1
	and $m10, $m9, 7
	brnz $m10, .Lattention_softmax_max_no_panel_step
	add $m3, $m3, $m8
.Lattention_softmax_max_no_panel_step:
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_max_pairs
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
	ldconst $a6, 0

	// Rewind the two panel streams. Normalization, F16 rounding, exponentiation,
	// denominator accumulation, and the output store are fused in one pass.
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	shl $m10, $m4, 5
	add $m2, $m2, $m10
	add $m3, $m3, $m10
	shr $m7, $m6, 1
	setzi $m9, 0
	brz $m7, .Lattention_softmax_norm_odd
.Lattention_softmax_norm_pairs:
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a2:3, $a4:B, $a0:1
	f32v2add $a2:3, $a5:B, $a2:3
	f32v2tof16 $a2, $a2:3
	f16v2exp $a2, $a2
	st32 $a2, $m2, $m15, 0
	f16v2tof32 $a2:3, $a2
	f32add $a6, $a6, $a2
	f32add $a6, $a6, $a3
	add $m2, $m2, 4
	add $m3, $m3, 4
	add $m9, $m9, 1
	and $m10, $m9, 7
	brnz $m10, .Lattention_softmax_norm_no_panel_step
	add $m2, $m2, $m8
	add $m3, $m3, $m8
.Lattention_softmax_norm_no_panel_step:
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_norm_pairs
.Lattention_softmax_norm_odd:
	and $m10, $m6, 1
	brz $m10, .Lattention_softmax_zero_padding
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a2:3, $a4:B, $a0:1
	f32v2add $a2:3, $a5:B, $a2:3
	// The high lane is outside the logical key block. Converting -65504 before
	// applying exp guarantees that the stored padding lane is zero.
	ldconst $a3, ATTENTION_MINIMUM_F32
	f32v2tof16 $a2, $a2:3
	f16v2exp $a2, $a2
	st32 $a2, $m2, $m15, 0
	f16v2tof32 $a2:3, $a2
	f32add $a6, $a6, $a2
	add $m2, $m2, 4
	add $m9, $m9, 1
	and $m10, $m9, 7
	brnz $m10, .Lattention_softmax_zero_padding
	add $m2, $m2, $m8
.Lattention_softmax_zero_padding:
	add $m7, $m6, 1
	shr $m7, $m7, 1
	setzi $m10, ATTENTION_PHYSICAL_KEY_PAIRS
	sub $m7, $m10, $m7
	brz $m7, .Lattention_softmax_store_state
.Lattention_softmax_zero_pairs:
	st32 $azero, $m2, $m15, 0
	add $m2, $m2, 4
	add $m9, $m9, 1
	and $m10, $m9, 7
	brnz $m10, .Lattention_softmax_zero_no_panel_step
	add $m2, $m2, $m8
.Lattention_softmax_zero_no_panel_step:
	sub $m7, $m7, 1
	brnz $m7, .Lattention_softmax_zero_pairs
.Lattention_softmax_store_state:
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m10, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m10, $m5, $m10
	add $m2, $m2, $m10
	shl $m10, $m4, 2
	add $m2, $m2, $m10
	st32 $a7, $m2, $m15, 0
	shl $m10, $m5, 2
	add $m2, $m2, $m10
	st32 $a6, $m2, $m15, 0
	add $m4, $m4, 6
	cmpult $m7, $m4, $m5
	brnz $m7, .Lattention_softmax_row
.Lattention_softmax_done:
	exitz $m15

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
	setzi $m2, ATTENTION_VALUE_PAIRS
	setzi $m3, 0
.Lattention_merge_initial_values:
	ld32 $a0, $m1, $m15, 0
	f16v2tof32 $a0:1, $a0
	st64 $a0:1, $m0, $m15, 0
	add $m0, $m0, 8
	add $m1, $m1, 4
	add $m3, $m3, 1
	and $m4, $m3, 7
	brnz $m4, .Lattention_merge_initial_no_panel_step
	add $m1, $m1, $m10
.Lattention_merge_initial_no_panel_step:
	sub $m2, $m2, 1
	brnz $m2, .Lattention_merge_initial_values
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
	setzi $m2, ATTENTION_VALUE_PAIRS
	setzi $m3, 0
.Lattention_merge_update_values:
	ld32 $a0, $m1, $m15, 0
	f16v2tof32 $a0:1, $a0
	ld64 $a2:3, $m0, $m15, 0
	f32v2mul $a2:3, $a5:B, $a2:3
	f32v2mul $a0:1, $a6:B, $a0:1
	f32v2add $a2:3, $a2:3, $a0:1
	st64 $a2:3, $m0, $m15, 0
	add $m0, $m0, 8
	add $m1, $m1, 4
	add $m3, $m3, 1
	and $m4, $m3, 7
	brnz $m4, .Lattention_merge_update_no_panel_step
	add $m1, $m1, $m10
.Lattention_merge_update_no_panel_step:
	sub $m2, $m2, 1
	brnz $m2, .Lattention_merge_update_values

.Lattention_merge_maybe_normalize:
	brz $m7, .Lattention_merge_next_row
	ld32 $a7, $m8, $m15, 1
	f32exp $a6, $azero
	f32div $a7, $a6, $a7
	ld32 $m0, $mvertex_base, $m15, 0
	setzi $m2, ATTENTION_PADDED_VALUE_DIMENSION * 4
	mul $m2, $m9, $m2
	add $m0, $m0, $m2
	setzi $m2, ATTENTION_VALUE_PAIRS
.Lattention_merge_normalize:
	ld64 $a0:1, $m0, $m15, 0
	f32v2mul $a0:1, $a7:B, $a0:1
	st64 $a0:1, $m0, $m15, 0
	add $m0, $m0, 8
	sub $m2, $m2, 1
	brnz $m2, .Lattention_merge_normalize
.Lattention_merge_next_row:
	add $m9, $m9, 6
	cmpult $m0, $m9, $m5
	brnz $m0, .Lattention_merge_row
.Lattention_merge_done:
	exitz $m15
