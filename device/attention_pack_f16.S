#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.macro ATTENTION_PACK_WRAPPER name, codelet
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -24
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m2, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 $m7, $m11, $m15, 5
	setzi $m0, \codelet
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size \name, .-\name
	.endm

	ATTENTION_PACK_WRAPPER ipu_stack_attention_pack_query_f16, __runCodelet_AttentionPackQueryF16
	ATTENTION_PACK_WRAPPER ipu_stack_attention_pack_key_f16, __runCodelet_AttentionPackKeyF16
	ATTENTION_PACK_WRAPPER ipu_stack_attention_pack_value_f16, __runCodelet_AttentionPackValueF16
