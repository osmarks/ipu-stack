#include "arch/gc_tile_defines.h"

#ifndef REARRANGE_CALL_SYMBOL
#define REARRANGE_CALL_SYMBOL ipu_stack_rearrange_row_major_to_amp_left_f16
#endif
#ifndef REARRANGE_LOGICAL_ROWS
#define REARRANGE_LOGICAL_ROWS 1
#endif
#ifndef REARRANGE_PHYSICAL_ROWS
#define REARRANGE_PHYSICAL_ROWS REARRANGE_LOGICAL_ROWS
#endif
#ifndef REARRANGE_LOGICAL_COLUMNS
#define REARRANGE_LOGICAL_COLUMNS 16
#endif
#ifndef REARRANGE_PHYSICAL_COLUMNS
#define REARRANGE_PHYSICAL_COLUMNS REARRANGE_LOGICAL_COLUMNS
#endif
#if REARRANGE_LOGICAL_COLUMNS % 2 != 0
#error "AMP-left F16 packing requires an even logical column count"
#endif
#if REARRANGE_PHYSICAL_COLUMNS % 16 != 0
#error "AMP-left F16 packing requires complete physical panels"
#endif
#if REARRANGE_PHYSICAL_COLUMNS < ((REARRANGE_LOGICAL_COLUMNS + 15) / 16) * 16
#error "AMP-left F16 packing requires enough physical columns"
#endif

#define REARRANGE_FULL_PANELS (REARRANGE_LOGICAL_COLUMNS / 16)
#define REARRANGE_TAIL_COLUMNS (REARRANGE_LOGICAL_COLUMNS % 16)
#define REARRANGE_USED_PANELS ((REARRANGE_LOGICAL_COLUMNS + 15) / 16)
#define REARRANGE_ZERO_PANELS ((REARRANGE_PHYSICAL_COLUMNS / 16) - REARRANGE_USED_PANELS)
#define REARRANGE_PANEL_ROW_ADVANCE ((REARRANGE_PHYSICAL_ROWS - 1) * 32)

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.REARRANGE_CALL_SYMBOL,"ax",@progbits
	.globl REARRANGE_CALL_SYMBOL
	.p2align 2
	.type REARRANGE_CALL_SYMBOL,@function
REARRANGE_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	setzi $m0, .Lrearrange_amp_left_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size REARRANGE_CALL_SYMBOL, .-REARRANGE_CALL_SYMBOL

	// AMP-left storage consists of row-major 16-column panels. Assign rows to
	// workers so every panel is written by all six contexts in bank-friendly
	// order. Complete pairs of F16 values move as one 32-bit word.
	.worker
	.p2align 3
.Lrearrange_amp_left_f16_worker:
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	get $m6, $WSR
	and $m6, $m6, CSR_W_WSR__CTXTID_M1__MASK
.Lrearrange_amp_left_f16_row:
	// Source rows are contiguous; destination rows advance through 16-column
	// panels with physicalRows * 32 bytes between panel starts.
	setzi $m2, REARRANGE_LOGICAL_COLUMNS
	mul $m9, $m6, $m2
	shl $m9, $m9, 1
	add $m9, $m0, $m9
	add $m10, $m6, 0
	shl $m10, $m10, 5
	add $m10, $m1, $m10

	setzi $m3, REARRANGE_LOGICAL_ROWS
	cmpult $m3, $m6, $m3
	brz $m3, .Lrearrange_amp_left_f16_zero_row

#if REARRANGE_FULL_PANELS > 0
	setzi $m7, REARRANGE_FULL_PANELS
.Lrearrange_amp_left_f16_full_panel:
	.rept 8
	ld32 $a0, $m9, $m15, 0
	st32 $a0, $m10, $m15, 0
	add $m9, $m9, 4
	add $m10, $m10, 4
	.endr
	add $m10, $m10, REARRANGE_PANEL_ROW_ADVANCE
	sub $m7, $m7, 1
	brnz $m7, .Lrearrange_amp_left_f16_full_panel
#endif

#if REARRANGE_TAIL_COLUMNS > 0
	.rept REARRANGE_TAIL_COLUMNS / 2
	ld32 $a0, $m9, $m15, 0
	st32 $a0, $m10, $m15, 0
	add $m9, $m9, 4
	add $m10, $m10, 4
	.endr
	zero $a0
	.rept (16 - REARRANGE_TAIL_COLUMNS) / 2
	st32 $a0, $m10, $m15, 0
	add $m10, $m10, 4
	.endr
	add $m10, $m10, REARRANGE_PANEL_ROW_ADVANCE
#endif

#if REARRANGE_ZERO_PANELS > 0
	setzi $m7, REARRANGE_ZERO_PANELS
.Lrearrange_amp_left_f16_zero_tail_panel:
	zero $a0
	.rept 8
	st32 $a0, $m10, $m15, 0
	add $m10, $m10, 4
	.endr
	add $m10, $m10, REARRANGE_PANEL_ROW_ADVANCE
	sub $m7, $m7, 1
	brnz $m7, .Lrearrange_amp_left_f16_zero_tail_panel
#endif
	bri .Lrearrange_amp_left_f16_next_row

.Lrearrange_amp_left_f16_zero_row:
	setzi $m7, REARRANGE_PHYSICAL_COLUMNS / 16
.Lrearrange_amp_left_f16_zero_row_panel:
	zero $a0
	.rept 8
	st32 $a0, $m10, $m15, 0
	add $m10, $m10, 4
	.endr
	add $m10, $m10, REARRANGE_PANEL_ROW_ADVANCE
	sub $m7, $m7, 1
	brnz $m7, .Lrearrange_amp_left_f16_zero_row_panel

.Lrearrange_amp_left_f16_next_row:
	add $m6, $m6, 6
	setzi $m2, REARRANGE_PHYSICAL_ROWS
	cmpult $m2, $m6, $m2
	brnz $m2, .Lrearrange_amp_left_f16_row
.Lrearrange_amp_left_f16_exit:
	exitz $m15
