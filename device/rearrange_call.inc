// Shared synchronous matrix loop for dense F16 packing workers.
// REARRANGE_ARGUMENTS follows the worker frame order.
	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.REARRANGE_CALL_SYMBOL,"ax",@progbits
	.globl REARRANGE_CALL_SYMBOL
	.p2align 2
	.type REARRANGE_CALL_SYMBOL,@function
REARRANGE_CALL_SYMBOL:
	.supervisor
	// Kernel ABI: m2 destination, m3 source, m4 logical rows,
	// m5 physical rows, m6 target order, m7 logical columns,
	// m8 physical columns, m9 matrices, m10 return address.
	add $m11, $m11, -16
.Lrearrange_matrix:
	.set rearrange_argument_index, 0
	.irp argument, REARRANGE_ARGUMENTS
	st32 \argument, $m11, $m15, rearrange_argument_index
	.set rearrange_argument_index, rearrange_argument_index + 1
	.endr
	setzi $m0, REARRANGE_WORKER
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	mul $m0, $m4, $m7
	shl $m0, $m0, 1
	add $m3, $m3, $m0
	mul $m0, $m5, $m8
	shl $m0, $m0, 1
	add $m2, $m2, $m0
	sub $m9, $m9, 1
	brnz $m9, .Lrearrange_matrix
	add $m11, $m11, 16
	br $m10
	.size REARRANGE_CALL_SYMBOL, .-REARRANGE_CALL_SYMBOL

