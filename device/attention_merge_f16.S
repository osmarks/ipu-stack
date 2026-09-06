#include "arch/gc_tile_defines.h"

#ifndef ATTENTION_VALUE_DIMENSION
#define ATTENTION_VALUE_DIMENSION 64
#endif
#ifndef ATTENTION_PADDED_VALUE_DIMENSION
#define ATTENTION_PADDED_VALUE_DIMENSION ATTENTION_VALUE_DIMENSION
#endif
#ifndef ATTENTION_KEY_BLOCK_COLUMNS
#define ATTENTION_KEY_BLOCK_COLUMNS 64
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
	.set ATTENTION_VALUE_FULL_PANELS, ATTENTION_VALUE_DIMENSION / 16
	.set ATTENTION_VALUE_TAIL_PAIRS, (ATTENTION_VALUE_DIMENSION % 16 + 1) / 2

	// Callable supervisor ABI for merge: m2 accumulator, m3 block values,
	// m4 softmax state, m5 initial-block flag, m6 final-block flag, m7 query rows, m10 return.
	.macro ATTENTION_MERGE_ENTRY name
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
	st32 $m7, $m11, $m15, 5
	setzi $m0, .Lattention_merge_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size \name, .-\name
	.endm

	ATTENTION_MERGE_ENTRY ATTENTION_MERGE_SYMBOL

	.macro ATTENTION_MERGE_INITIAL_PAIR
	ld32step $a0, $mzero, $m1+=, 1
	f16v2tof32 $a0:1, $a0
	f32v2mul $a0:1, $a6:B, $a0:1
	st64step $a0:1, $mzero, $m0+=, 1
	.endm

	.macro ATTENTION_MERGE_UPDATE_PAIR
	ld32step $a0, $mzero, $m1+=, 1
	f16v2tof32 $a0:1, $a0
	{ ld64 $a2:3, $m0, $m15, 0
	  f32v2mul $a0:1, $a6:B, $a0:1 }
	f32v2mul $a2:3, $a5:B, $a2:3
	f32v2add $a2:3, $a2:3, $a0:1
	st64step $a2:3, $mzero, $m0+=, 1
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
	ldconst $a6, 0x3f800000
	brz $m7, .Lattention_merge_initial_ready
	f32oox $a6, $a1
.Lattention_merge_initial_ready:
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
	bri .Lattention_merge_next_row

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
	// Normalize the final update through its coefficients, avoiding a second
	// pass over the output. Persistent sums and coefficients remain FP32.
	brz $m7, .Lattention_merge_update_ready
	f32oox $a7, $a3
	f32mul $a5, $a5, $a7
	f32mul $a6, $a6, $a7
.Lattention_merge_update_ready:
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

.Lattention_merge_next_row:
	add $m9, $m9, 6
	cmpult $m0, $m9, $m5
	brnz $m0, .Lattention_merge_row
.Lattention_merge_done:
	exitz $m15
