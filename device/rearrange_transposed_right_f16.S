#include "arch/gc_tile_defines.h"

#ifndef REARRANGE_CALL_SYMBOL
#define REARRANGE_CALL_SYMBOL ipu_stack_rearrange_row_major_to_amp_f16_transposed_right
#endif

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.REARRANGE_CALL_SYMBOL,"ax",@progbits
	.globl REARRANGE_CALL_SYMBOL
	.p2align 2
	.type REARRANGE_CALL_SYMBOL,@function
REARRANGE_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	setzi $m0, .Lrearrange_transposed_right_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size REARRANGE_CALL_SYMBOL, .-REARRANGE_CALL_SYMBOL

	// Specialized 64x16 row-major to transposed coefficient-panel packing. Each
	// input row is one contiguous 32-byte load channel; ld*putcs destinations
	// perform the final AMP coefficient permutation at consumption time.
	.worker
	.p2align 3
.Lrearrange_transposed_right_f16_worker:
	ld32 $m3, $mvertex_base, $m15, 0
	ld32 $m2, $mvertex_base, $m15, 1
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
.Lrearrange_transposed_right_f16_row:
	shl $m5, $m4, 5
	add $m6, $m3, $m5
	shr $m5, $m4, 4
	shl $m5, $m5, 9
	add $m7, $m2, $m5
	and $m5, $m4, 15
	shl $m5, $m5, 5
	add $m7, $m7, $m5
	// Staging buffers guarantee eight-byte alignment; they are not required to
	// be 16-byte aligned, so use four wide loads instead of ld128.
	ld64 $a0:1, $m6, $m15, 0
	ld64 $a2:3, $m6, $m15, 1
	ld64 $a4:5, $m6, $m15, 2
	ld64 $a6:7, $m6, $m15, 3
	st64 $a0:1, $m7, $m15, 0
	st64 $a2:3, $m7, $m15, 1
	st64 $a4:5, $m7, $m15, 2
	st64 $a6:7, $m7, $m15, 3
	add $m4, $m4, 6
	setzi $m5, 64
	cmpult $m5, $m4, $m5
	brnz $m5, .Lrearrange_transposed_right_f16_row
	exitz $m15
