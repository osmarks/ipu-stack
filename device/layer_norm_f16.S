#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set ENABLE_STOCHASTIC_ROUNDING, 1 << CSR_S_FP_ICTL__ESR__SHIFT
	.section .text.ipu_stack_layer_norm_affine_f16,"ax",@progbits
	.globl ipu_stack_layer_norm_affine_f16
	.p2align 2
	.type ipu_stack_layer_norm_affine_f16,@function
ipu_stack_layer_norm_affine_f16:
	.supervisor
	add $m11, $m11, -32
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m2, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 $m7, $m11, $m15, 5
	get $m8, $FP_ICTL
	st32 $m8, $m11, $m15, 6
	setzi $m0, ENABLE_STOCHASTIC_ROUNDING
	or $m9, $m8, $m0
	put $FP_ICTL, $m9
	setzi $m0, __runCodelet_LayerNormAffineF16
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	ld32 $m8, $m11, $m15, 6
	put $FP_ICTL, $m8
	add $m11, $m11, 32
	br $m10
	.size ipu_stack_layer_norm_affine_f16, .-ipu_stack_layer_norm_affine_f16

	.section .text.ipu_stack_add_layer_norm_affine_f16,"ax",@progbits
	.globl ipu_stack_add_layer_norm_affine_f16
	.p2align 2
	.type ipu_stack_add_layer_norm_affine_f16,@function
ipu_stack_add_layer_norm_affine_f16:
	.supervisor
	add $m11, $m11, -40
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m2, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 $m7, $m11, $m15, 5
	st32 $m8, $m11, $m15, 6
	get $m9, $FP_ICTL
	st32 $m9, $m11, $m15, 7
	setzi $m0, ENABLE_STOCHASTIC_ROUNDING
	or $m0, $m9, $m0
	put $FP_ICTL, $m0
	setzi $m0, __runCodelet_AddLayerNormAffineF16
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	ld32 $m9, $m11, $m15, 7
	put $FP_ICTL, $m9
	add $m11, $m11, 40
	br $m10
	.size ipu_stack_add_layer_norm_affine_f16, .-ipu_stack_add_layer_norm_affine_f16
