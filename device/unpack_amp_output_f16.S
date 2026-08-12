	#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	#ifndef UNPACK_CALL_SYMBOL
	#define UNPACK_CALL_SYMBOL ipu_stack_rearrange_amp_output_to_row_major_f16
	#endif
	#ifndef UNPACK_COLUMNS
	#define UNPACK_COLUMNS 16
	#endif
	#if UNPACK_COLUMNS % 16
	#error "UNPACK_COLUMNS must be a multiple of 16"
	#endif

	.section .text.UNPACK_CALL_SYMBOL,"ax",@progbits
	.globl UNPACK_CALL_SYMBOL
	.p2align 2
	.type UNPACK_CALL_SYMBOL,@function
UNPACK_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -32
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	setzi $m0, .Lunpack_amp_output_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 32
	br $m10
	.size UNPACK_CALL_SYMBOL, .-UNPACK_CALL_SYMBOL

	// AMP output stores each 16-column group in the fixed pair order
	// 0, 4, 1, 5, 2, 6, 3, 7. Assigning whole rows to workers makes both
	// source and destination addressing affine and avoids per-element index
	// arithmetic. ABI fields are source, destination, and row count.
	.macro UNPACK_PAIR source_offset, destination_offset
	ld32 $a0, $m3, $m15, \source_offset
	st32 $a0, $m2, $m15, \destination_offset
	.endm

	.worker
	.p2align 3
.Lunpack_amp_output_f16_worker:
	ld32 $m3, $mvertex_base, $m15, 0
	ld32 $m2, $mvertex_base, $m15, 1
	ld32 $m8, $mvertex_base, $m15, 2
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m0, $m4, $m8
	brz $m0, .Lunpack_amp_output_f16_done
	shl $m5, $m4, 5
	add $m3, $m3, $m5
	setzi $m5, UNPACK_COLUMNS * 2
	mul $m5, $m4, $m5
	add $m2, $m2, $m5
	shl $m7, $m8, 5
.Lunpack_amp_output_f16_row:
	add $m9, $m3, 0
	add $m6, $m2, 0
	setzi $m5, UNPACK_COLUMNS / 16
.Lunpack_amp_output_f16_group:
	UNPACK_PAIR 0, 0
	UNPACK_PAIR 1, 4
	UNPACK_PAIR 2, 1
	UNPACK_PAIR 3, 5
	UNPACK_PAIR 4, 2
	UNPACK_PAIR 5, 6
	UNPACK_PAIR 6, 3
	UNPACK_PAIR 7, 7
	add $m3, $m3, $m7
	add $m2, $m2, 32
	sub $m5, $m5, 1
	brnz $m5, .Lunpack_amp_output_f16_group
	add $m3, $m9, 192
	add $m2, $m6, UNPACK_COLUMNS * 12
	add $m4, $m4, 6
	cmpult $m0, $m4, $m8
	brnz $m0, .Lunpack_amp_output_f16_row
.Lunpack_amp_output_f16_done:
	exitz $m15
