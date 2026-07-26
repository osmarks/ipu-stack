#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
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
	setzi $m6, REBLOCK_ROW_MASK
	and $m6, $m7, $m6
	cmpult $m6, $m4, $m6
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

	.macro REBLOCK_WRAPPER name, worker
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -24
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m5, $m11, $m15, 2
	st32 $m6, $m11, $m15, 3
	st32 $m7, $m11, $m15, 4
	setzi $m0, \worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 24
	br $m10
	.size \name, .-\name
	.endm

	REBLOCK_WRAPPER ipu_stack_reblock_f16_a16_to_a16, .La16_worker
	REBLOCK_WRAPPER ipu_stack_reblock_f16_c16_to_a16, .Lc16_worker

	.macro REBLOCK_WORKER_SETUP
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m5, $mvertex_base, $m15, 2
	ld32 $m6, $mvertex_base, $m15, 3
	ld32 $m7, $mvertex_base, $m15, 4
	REBLOCK_SETUP
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	.endm

	.worker
	.p2align 3
.La16_worker:
	REBLOCK_WORKER_SETUP
.La16_row:
	REBLOCK_ROW_POINTERS .La16_done
	mov $m10, $m4
	zero $m4
.La16_panel:
	shr $m6, $m7, 16
	cmpult $m6, $m4, $m6
	brz $m6, .La16_row_done
	COPY_A16_PANEL
	add $m4, $m4, 1
	bri .La16_panel
.La16_row_done:
	mov $m4, $m10
	add $m4, $m4, 6
	bri .La16_row
.La16_done:
	exitz $m15

	.worker
	.p2align 3
.Lc16_worker:
	REBLOCK_WORKER_SETUP
.Lc16_row:
	REBLOCK_ROW_POINTERS .Lc16_done
	mov $m10, $m4
	zero $m4
.Lc16_panel:
	shr $m6, $m7, 16
	cmpult $m6, $m4, $m6
	brz $m6, .Lc16_row_done
	COPY_C16_PANEL
	add $m4, $m4, 1
	bri .Lc16_panel
.Lc16_row_done:
	mov $m4, $m10
	add $m4, $m4, 6
	bri .Lc16_row
.Lc16_done:
	exitz $m15
