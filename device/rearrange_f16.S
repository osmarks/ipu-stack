	#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	#ifndef REARRANGE_CALL_SYMBOL
	#define REARRANGE_CALL_SYMBOL ipu_stack_rearrange_row_major_to_amp_f16
	#endif
	#ifndef REARRANGE_CODELET_SYMBOL
	#define REARRANGE_CODELET_SYMBOL __runCodelet_RearrangeRowMajorToAmpF16
	#endif

	.section .text.REARRANGE_CALL_SYMBOL,"ax",@progbits
	.globl REARRANGE_CALL_SYMBOL
	.p2align 2
	.type REARRANGE_CALL_SYMBOL,@function
REARRANGE_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -32
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	setzi $m0, REARRANGE_CODELET_SYMBOL
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 32
	br $m10
	.size REARRANGE_CALL_SYMBOL, .-REARRANGE_CALL_SYMBOL
