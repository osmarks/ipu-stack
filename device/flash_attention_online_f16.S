#include "arch/gc_tile_defines.h"

#ifndef ATTENTION_CALL_SYMBOL
#define ATTENTION_CALL_SYMBOL ipu_stack_flash_attention_online_f16
#endif
#ifndef ATTENTION_CODELET_SYMBOL
#define ATTENTION_CODELET_SYMBOL __runCodelet_FlashAttentionOnlineF16
#endif

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.ATTENTION_CALL_SYMBOL,"ax",@progbits
	.globl ATTENTION_CALL_SYMBOL
	.p2align 2
	.type ATTENTION_CALL_SYMBOL,@function
ATTENTION_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m4, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m2, $m11, $m15, 3
	setzi $m0, ATTENTION_CODELET_SYMBOL
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size ATTENTION_CALL_SYMBOL, .-ATTENTION_CALL_SYMBOL
