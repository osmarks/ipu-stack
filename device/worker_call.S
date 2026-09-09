#include "arch/gc_tile_defines.h"

// WORKER_ARGUMENTS lists supervisor registers in C++ vertex-field order.
// All wrappers use the same aligned stack frame and synchronous worker launch.
	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.WORKER_CALL_SYMBOL,"ax",@progbits
	.globl WORKER_CALL_SYMBOL
	.p2align 2
	.type WORKER_CALL_SYMBOL,@function
WORKER_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -WORKER_FRAME_BYTES
	.set argument_index, 0
	.irp argument, WORKER_ARGUMENTS
	st32 \argument, $m11, $m15, argument_index
	.set argument_index, argument_index + 1
	.endr
#ifdef WORKER_SELECT_CODELET
#include WORKER_SELECT_CODELET
#else
	setzi $m0, WORKER_CODELET_SYMBOL
#endif
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, WORKER_FRAME_BYTES
	br $m10
	.size WORKER_CALL_SYMBOL, .-WORKER_CALL_SYMBOL
