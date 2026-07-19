#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set ATTENTION_WORKER_STACK_BYTES, 0xe0
	.set ENABLE_STOCHASTIC_ROUNDING, 1 << CSR_S_FP_ICTL__ESR__SHIFT

	.globl __worker_stack_max_growth_plus_one
	.set __worker_stack_max_growth_plus_one, ATTENTION_WORKER_STACK_BYTES + 1

	.section .text._stackoverflow,"ax",@progbits
	.globl _stackoverflow
	.p2align 2
	.type _stackoverflow,@function
_stackoverflow:
	.worker
	trap 0
	.size _stackoverflow, .-_stackoverflow

	.macro ATTENTION_SOFTMAX_WRAPPER name, codelet
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m2, $m11, $m15, 2
	setzi $m0, \codelet
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size \name, .-\name
	.endm

	ATTENTION_SOFTMAX_WRAPPER ipu_stack_attention_softmax_small_query_small_key_f16, __runCodelet_AttentionSoftmaxSmallQuerySmallKeyF16
	ATTENTION_SOFTMAX_WRAPPER ipu_stack_attention_softmax_small_query_large_key_f16, __runCodelet_AttentionSoftmaxSmallQueryLargeKeyF16
	ATTENTION_SOFTMAX_WRAPPER ipu_stack_attention_softmax_large_query_small_key_f16, __runCodelet_AttentionSoftmaxLargeQuerySmallKeyF16
	ATTENTION_SOFTMAX_WRAPPER ipu_stack_attention_softmax_large_query_large_key_f16, __runCodelet_AttentionSoftmaxLargeQueryLargeKeyF16

	.macro ATTENTION_MERGE_WRAPPER name, codelet
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -32
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m2, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	setzi $m0, \codelet
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 32
	br $m10
	.size \name, .-\name
	.endm

	ATTENTION_MERGE_WRAPPER ipu_stack_attention_merge_small_query_f16, __runCodelet_AttentionMergeSmallQueryF16
	ATTENTION_MERGE_WRAPPER ipu_stack_attention_merge_large_query_f16, __runCodelet_AttentionMergeLargeQueryF16

	.section .text.ipu_stack_attention_f32_to_f16,"ax",@progbits
	.globl ipu_stack_attention_f32_to_f16
	.p2align 2
	.type ipu_stack_attention_f32_to_f16,@function
ipu_stack_attention_f32_to_f16:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	get $m8, $FP_ICTL
	st32 $m8, $m11, $m15, 3
	setzi $m0, ENABLE_STOCHASTIC_ROUNDING
	or $m8, $m8, $m0
	put $FP_ICTL, $m8
	setzi $m0, .Lattention_f32_to_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	ld32 $m8, $m11, $m15, 3
	put $FP_ICTL, $m8
	add $m11, $m11, 16
	br $m10
	.size ipu_stack_attention_f32_to_f16, .-ipu_stack_attention_f32_to_f16

	.worker
	.p2align 3
.Lattention_f32_to_f16_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m5, $mvertex_base, $m15, 2
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	setzi $m6, 6
	mul $m7, $m4, 4
	add $m2, $m2, $m7
	shl $m7, $m7, 1
	add $m3, $m3, $m7
	setzi $m0, .Lattention_f32_to_f16_loop
.Lattention_f32_to_f16_loop:
	cmpult $m8, $m4, $m5
	brz $m8, .Lattention_f32_to_f16_done
	ld64 $a0:1, $m3, $m15, 0
	f32v2tof16 $a0, $a0:1
	st32 $a0, $m2, $m15, 0
	add $m2, $m2, 24
	add $m3, $m3, 48
	add $m4, $m4, $m6
	br $m0
.Lattention_f32_to_f16_done:
	exitz $m15
