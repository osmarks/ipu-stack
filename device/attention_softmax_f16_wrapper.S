#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

#ifndef ATTENTION_SOFTMAX_SMALL_QUERY_SMALL_KEY_SYMBOL
#define ATTENTION_SOFTMAX_SMALL_QUERY_SMALL_KEY_SYMBOL ipu_stack_attention_softmax_small_query_small_key_f16
#endif
#ifndef ATTENTION_SOFTMAX_LARGE_QUERY_SMALL_KEY_SYMBOL
#define ATTENTION_SOFTMAX_LARGE_QUERY_SMALL_KEY_SYMBOL ipu_stack_attention_softmax_large_query_small_key_f16
#endif
#ifndef ATTENTION_SOFTMAX_SMALL_QUERY_LARGE_KEY_SYMBOL
#define ATTENTION_SOFTMAX_SMALL_QUERY_LARGE_KEY_SYMBOL ipu_stack_attention_softmax_small_query_large_key_f16
#endif
#ifndef ATTENTION_SOFTMAX_LARGE_QUERY_LARGE_KEY_SYMBOL
#define ATTENTION_SOFTMAX_LARGE_QUERY_LARGE_KEY_SYMBOL ipu_stack_attention_softmax_large_query_large_key_f16
#endif

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

#ifndef ATTENTION_USE_ASSEMBLY_LARGE_KEY
	ATTENTION_SOFTMAX_WRAPPER ATTENTION_SOFTMAX_SMALL_QUERY_LARGE_KEY_SYMBOL, __runCodelet_AttentionSoftmaxSmallQueryLargeKeyF16
	ATTENTION_SOFTMAX_WRAPPER ATTENTION_SOFTMAX_LARGE_QUERY_LARGE_KEY_SYMBOL, __runCodelet_AttentionSoftmaxLargeQueryLargeKeyF16
#endif
#ifndef ATTENTION_USE_ASSEMBLY_SMALL_KEY
	ATTENTION_SOFTMAX_WRAPPER ATTENTION_SOFTMAX_SMALL_QUERY_SMALL_KEY_SYMBOL, __runCodelet_AttentionSoftmaxSmallQuerySmallKeyF16
	ATTENTION_SOFTMAX_WRAPPER ATTENTION_SOFTMAX_LARGE_QUERY_SMALL_KEY_SYMBOL, __runCodelet_AttentionSoftmaxLargeQuerySmallKeyF16
#endif
