#include "arch/gc_tile_defines.h"

#ifndef REARRANGE_CALL_SYMBOL
#define REARRANGE_CALL_SYMBOL ipu_stack_rearrange_row_major_to_block_major_f16_64x16
#endif
#ifndef REARRANGE_PHYSICAL_COLUMNS
#define REARRANGE_PHYSICAL_COLUMNS 16
#endif
#ifndef REARRANGE_PHYSICAL_ROWS
#define REARRANGE_PHYSICAL_ROWS 64
#endif

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
	// m8 physical columns, m10 return address.
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m7, $m11, $m15, 3
	setzi $m0, .Lrearrange_block_major_f16_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size REARRANGE_CALL_SYMBOL, .-REARRANGE_CALL_SYMBOL

	// Transpose each 2x2 F16 cell while converting a row-major 64xN matrix
	// into 64x16-panel block-major coefficient order. Each 64-bit source load
	// covers four adjacent columns from one row. sort4x16lo/hi pair the same
	// column from two adjacent rows, which is the natural source order expected
	// by the GEMM supervisor's ld*putcs coefficient routing.
	.worker
	.p2align 3
.Lrearrange_block_major_f16_worker:
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	ld32 $m2, $mvertex_base, $m15, 2
	ld32 $m3, $mvertex_base, $m15, 3
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	shl $m4, $m4, 1
.Lrearrange_block_major_f16_row_pair:
	// m6/m7 are the two source rows. They are only dereferenced when the
	// corresponding logical row exists.
	mul $m5, $m4, $m3
	shl $m5, $m5, 1
	add $m6, $m0, $m5
	shl $m5, $m3, 1
	add $m7, $m6, $m5
	add $m5, $m4, 1
	cmpult $m8, $m5, $m2

	zero $m9
.Lrearrange_block_major_f16_columns:
	// A 64x16 column block consists of four 16x16 coefficient panels.
	// Select the column block, the 16-row panel within it, and the row and
	// column offsets within that panel.
	shr $m5, $m9, 4
	shl $m5, $m5, 11
	add $m10, $m1, $m5
	shr $m5, $m4, 4
	shl $m5, $m5, 9
	add $m10, $m10, $m5
	and $m5, $m4, 15
	shl $m5, $m5, 1
	add $m10, $m10, $m5
	and $m5, $m9, 15
	shl $m5, $m5, 5
	add $m10, $m10, $m5

	cmpult $m5, $m4, $m2
	brz $m5, .Lrearrange_block_major_f16_zero_rows
	cmpult $m5, $m9, $m3
	brz $m5, .Lrearrange_block_major_f16_zero_rows
	ld64 $a0:1, $m6, $m15, 0
	brz $m8, .Lrearrange_block_major_f16_zero_second_row
	ld64 $a2:3, $m7, $m15, 0
	bri .Lrearrange_block_major_f16_transpose
.Lrearrange_block_major_f16_zero_second_row:
	mov $a2:3, $azeros
	bri .Lrearrange_block_major_f16_transpose
.Lrearrange_block_major_f16_zero_rows:
	mov $a0:1, $azeros
	mov $a2:3, $azeros
.Lrearrange_block_major_f16_transpose:
	sort4x16lo $a4, $a0, $a2
	sort4x16hi $a5, $a0, $a2
	sort4x16lo $a6, $a1, $a3
	sort4x16hi $a7, $a1, $a3
	st32 $a4, $m10, $m15, 0
	st32 $a5, $m10, $m15, 8
	st32 $a6, $m10, $m15, 16
	st32 $a7, $m10, $m15, 24
	add $m6, $m6, 8
	add $m7, $m7, 8
	add $m9, $m9, 4
	setzi $m5, REARRANGE_PHYSICAL_COLUMNS
	cmpult $m5, $m9, $m5
	brnz $m5, .Lrearrange_block_major_f16_columns

	add $m4, $m4, 12
	setzi $m5, REARRANGE_PHYSICAL_ROWS
	cmpult $m5, $m4, $m5
	brnz $m5, .Lrearrange_block_major_f16_row_pair
	exitz $m15
