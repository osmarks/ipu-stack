#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.section .text.ipu_stack_check_finite_f16,"ax",@progbits
	.globl ipu_stack_check_finite_f16
	.p2align 2
	.type ipu_stack_check_finite_f16,@function
ipu_stack_check_finite_f16:
	.supervisor
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	setzi $m0, __runCodelet_CheckFiniteF16
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size ipu_stack_check_finite_f16, .-ipu_stack_check_finite_f16
