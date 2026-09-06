#include "arch/gc_tile_defines.h"

// Transposed AMP stores each 16-row panel column-major. Transpose 2x2
// cells with word loads and lane shuffles; never perform FP arithmetic.

	.text
	.allow_optimizations
	.section .text.UNPACK_CALL_SYMBOL,"ax",@progbits
	.globl UNPACK_CALL_SYMBOL
	.p2align 2
	.type UNPACK_CALL_SYMBOL,@function
UNPACK_CALL_SYMBOL:
	.supervisor
	add $m11, $m11, -16
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	setzi $m0, .Lunpack_worker
	runall $m0, $m11, 0
	sync TEXCH_SYNCZONE_LOCAL
	add $m11, $m11, 16
	br $m10
	.size UNPACK_CALL_SYMBOL, .-UNPACK_CALL_SYMBOL

	.worker
	.p2align 3
.Lunpack_worker:
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	ld32 $m3, $mvertex_base, $m15, 2
	get $m2, $WSR
	and $m2, $m2, CSR_W_WSR__CTXTID_M1__MASK
	shl $m2, $m2, 1
	brz $m3, .Lunpack_done
.Lunpack_matrix:
	mov $m4, $m2
.Lunpack_rows:
	setzi $m5, UNPACK_PHYSICAL_ROWS
	cmpult $m5, $m4, $m5
	brz $m5, .Lunpack_next_matrix
	shr $m5, $m4, 4
	ldconst $m6, UNPACK_PHYSICAL_COLUMNS * 32
	mul $m5, $m5, $m6
	add $m6, $m0, $m5
	and $m5, $m4, 15
	shl $m5, $m5, 1
	add $m6, $m6, $m5
	ldconst $m5, UNPACK_PHYSICAL_COLUMNS * 2
	mul $m7, $m4, $m5
	add $m7, $m1, $m7
	add $m8, $m7, $m5
	zero $m9
.Lunpack_columns:
	mov $a0:1, $azeros
#if UNPACK_LOGICAL_ROWS < UNPACK_PHYSICAL_ROWS
	setzi $m5, UNPACK_LOGICAL_ROWS
	cmpult $m5, $m4, $m5
	brz $m5, .Lunpack_store
#endif
#if UNPACK_LOGICAL_COLUMNS < UNPACK_PHYSICAL_COLUMNS
	setzi $m5, UNPACK_LOGICAL_COLUMNS
	cmpult $m5, $m9, $m5
	brz $m5, .Lunpack_store
#endif
	ld32 $a2, $m6, $m15, 0
	mov $a3, $azero
#if UNPACK_LOGICAL_COLUMNS % 2
	setzi $m5, UNPACK_LOGICAL_COLUMNS - 1
	cmpult $m5, $m9, $m5
	brz $m5, .Lunpack_shuffle
#endif
	ld32 $a3, $m6, $m15, 8
.Lunpack_shuffle:
	sort4x16lo $a0, $a2, $a3
	sort4x16hi $a1, $a2, $a3
#if UNPACK_LOGICAL_ROWS % 2
	setzi $m5, UNPACK_LOGICAL_ROWS - 1
	cmpeq $m5, $m4, $m5
	brz $m5, .Lunpack_store
	mov $a1, $azero
#endif
.Lunpack_store:
	st32 $a0, $m7, $m15, 0
	st32 $a1, $m8, $m15, 0
	add $m6, $m6, 64
	add $m7, $m7, 4
	add $m8, $m8, 4
	add $m9, $m9, 2
	setzi $m5, UNPACK_PHYSICAL_COLUMNS
	cmpult $m5, $m9, $m5
	brnz $m5, .Lunpack_columns
	add $m4, $m4, 12
	bri .Lunpack_rows
.Lunpack_next_matrix:
	ldconst $m5, UNPACK_PHYSICAL_ROWS * UNPACK_PHYSICAL_COLUMNS * 2
	add $m0, $m0, $m5
	add $m1, $m1, $m5
	add $m3, $m3, -1
	brnz $m3, .Lunpack_matrix
.Lunpack_done:
	exitz $m15
