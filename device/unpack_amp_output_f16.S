	#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	#ifndef UNPACK_CALL_SYMBOL
	#define UNPACK_CALL_SYMBOL ipu_stack_rearrange_amp_output_to_row_major_f16
	#endif
	#ifndef UNPACK_CODELET_SYMBOL
	#define UNPACK_CODELET_SYMBOL __runCodelet_RearrangeAmpOutputToRowMajorF16
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
	setzi $m0, UNPACK_CODELET_SYMBOL
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 32
	br $m10
	.size UNPACK_CALL_SYMBOL, .-UNPACK_CALL_SYMBOL
