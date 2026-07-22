#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set REBLOCK_ROW_BITS, 10
	.set REBLOCK_ROW_MASK, (1 << REBLOCK_ROW_BITS) - 1

	.macro REBLOCK_SETUP
	add $m0, $m3, 0
	add $m1, $m2, 0
	add $m2, $m5, 0
	shr $m3, $m2, REBLOCK_ROW_BITS
	setzi $m9, REBLOCK_ROW_MASK
	and $m2, $m2, $m9
	mul $m2, $m2, 32
	mul $m3, $m3, 32
	add $m8, $m6, 0
	zero $m4
	.endm

	.macro REBLOCK_ROW_POINTERS done
	cmpult $m6, $m4, $m7
	brz $m6, \done
	setzi $m9, REBLOCK_ROW_MASK
	add $m6, $m8, 0
	shr $m5, $m6, REBLOCK_ROW_BITS
	and $m6, $m6, $m9
	add $m9, $m6, $m4
	mul $m9, $m9, 32
	add $m9, $m0, $m9
	add $m5, $m5, $m4
	mul $m5, $m5, 32
	add $m5, $m1, $m5
	.endm

	.macro COPY_A16_WORD offset
	ld32 $m6, $m9, $m15, \offset
	st32 $m6, $m5, $m15, \offset
	.endm

	.macro COPY_A16_PANEL
	COPY_A16_WORD 0
	COPY_A16_WORD 1
	COPY_A16_WORD 2
	COPY_A16_WORD 3
	COPY_A16_WORD 4
	COPY_A16_WORD 5
	COPY_A16_WORD 6
	COPY_A16_WORD 7
	add $m9, $m9, $m2
	add $m5, $m5, $m3
	.endm

	.macro COPY_C16_PANEL
	ld32 $m6, $m9, $m15, 0
	st32 $m6, $m5, $m15, 0
	ld32 $m6, $m9, $m15, 1
	st32 $m6, $m5, $m15, 4
	ld32 $m6, $m9, $m15, 2
	st32 $m6, $m5, $m15, 1
	ld32 $m6, $m9, $m15, 3
	st32 $m6, $m5, $m15, 5
	ld32 $m6, $m9, $m15, 4
	st32 $m6, $m5, $m15, 2
	ld32 $m6, $m9, $m15, 5
	st32 $m6, $m5, $m15, 6
	ld32 $m6, $m9, $m15, 6
	st32 $m6, $m5, $m15, 3
	ld32 $m6, $m9, $m15, 7
	st32 $m6, $m5, $m15, 7
	add $m9, $m9, $m2
	add $m5, $m5, $m3
	.endm

	.section .text.ipu_stack_reblock_f16_a16_to_a16,"ax",@progbits
	.globl ipu_stack_reblock_f16_a16_to_a16
	.p2align 2
	.type ipu_stack_reblock_f16_a16_to_a16,@function
ipu_stack_reblock_f16_a16_to_a16:
	.supervisor
	REBLOCK_SETUP
.La16_row:
	REBLOCK_ROW_POINTERS .La16_done
	COPY_A16_PANEL
	COPY_A16_PANEL
	COPY_A16_PANEL
	COPY_A16_PANEL
	add $m4, $m4, 1
	bri .La16_row
.La16_done:
	br $m10
	.size ipu_stack_reblock_f16_a16_to_a16, .-ipu_stack_reblock_f16_a16_to_a16

	.section .text.ipu_stack_reblock_f16_c16_to_a16,"ax",@progbits
	.globl ipu_stack_reblock_f16_c16_to_a16
	.p2align 2
	.type ipu_stack_reblock_f16_c16_to_a16,@function
ipu_stack_reblock_f16_c16_to_a16:
	.supervisor
	REBLOCK_SETUP
.Lc16_row:
	REBLOCK_ROW_POINTERS .Lc16_done
	COPY_C16_PANEL
	COPY_C16_PANEL
	COPY_C16_PANEL
	COPY_C16_PANEL
	add $m4, $m4, 1
	bri .Lc16_row
.Lc16_done:
	br $m10
	.size ipu_stack_reblock_f16_c16_to_a16, .-ipu_stack_reblock_f16_c16_to_a16
