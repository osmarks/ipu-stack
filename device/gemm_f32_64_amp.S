#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set WORKER_ARGUMENTS, 0x50000
	.set BLOCK_DIMENSION, 64
	.set INNER_MICRO_DIMENSION, 8
	.set COLUMN_MICRO_DIMENSION, 16
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	// The SDK's supervisor CR header does not expose this shift to assembly.
	.set CLEAR_CWEI, 1 << 2
	.set CLEAR_AACC, 1 << CSR_W_FP_CLR__ZAACC__SHIFT

	.section .text.ipu_stack_copy_4096_u32,"ax",@progbits
	.globl ipu_stack_copy_4096_u32
	.p2align 2
	.type ipu_stack_copy_4096_u32,@function
ipu_stack_copy_4096_u32:
	.supervisor
	setzi $m4, BLOCK_DIMENSION * BLOCK_DIMENSION
.Lcopy_word:
	ld32 $m0, $m3, $m15, 0
	st32 $m0, $m2, $m15, 0
	add $m3, $m3, 4
	add $m2, $m2, 4
	sub $m4, $m4, 1
	brnz $m4, .Lcopy_word
	br $m10
	.size ipu_stack_copy_4096_u32, .-ipu_stack_copy_4096_u32

	.macro LOAD_AMP16_WEIGHTS
	ld64putcs 0
	ld64putcs 1
	ld64putcs 2
	ld64putcs 3
	ld64putcs 4
	ld64putcs 5
	ld64putcs 6
	ld64putcs 7
	ld64putcs 32
	ld64putcs 33
	ld64putcs 34
	ld64putcs 35
	ld64putcs 36
	ld64putcs 37
	ld64putcs 38
	ld64putcs 39
	ld64putcs 8
	ld64putcs 9
	ld64putcs 10
	ld64putcs 11
	ld64putcs 12
	ld64putcs 13
	ld64putcs 14
	ld64putcs 15
	ld64putcs 40
	ld64putcs 41
	ld64putcs 42
	ld64putcs 43
	ld64putcs 44
	ld64putcs 45
	ld64putcs 46
	ld64putcs 47
	ld64putcs 16
	ld64putcs 17
	ld64putcs 18
	ld64putcs 19
	ld64putcs 20
	ld64putcs 21
	ld64putcs 22
	ld64putcs 23
	ld64putcs 48
	ld64putcs 49
	ld64putcs 50
	ld64putcs 51
	ld64putcs 52
	ld64putcs 53
	ld64putcs 54
	ld64putcs 55
	ld64putcs 24
	ld64putcs 25
	ld64putcs 26
	ld64putcs 27
	ld64putcs 28
	ld64putcs 29
	ld64putcs 30
	ld64putcs 31
	ld64putcs 56
	ld64putcs 57
	ld64putcs 58
	ld64putcs 59
	ld64putcs 60
	ld64putcs 61
	ld64putcs 62
	ld64putcs 63
	.endm

	.macro GEMM_SUPERVISOR name, clear_output
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	.if \clear_output
	mov $m5, $m2
	setzi $m6, BLOCK_DIMENSION * BLOCK_DIMENSION
.Lzero_\@:
	st32 $m15, $m5, $m15, 0
	add $m5, $m5, 4
	sub $m6, $m6, 1
	brnz $m6, .Lzero_\@
	.endif
	setzi $m5, CLEAR_CWEI
	put $CR, $m5
	put $CCCSLOAD, $m4
	sync SYNC_COMPUTE_SET
	zero $m6
.Lcolumn_group_\@:
	zero $m7
.Linner_group_\@:
	LOAD_AMP16_WEIGHTS
	setzi $m5, WORKER_ARGUMENTS
	shl $m8, $m6, 6
	add $m8, $m2, $m8
	st32 $m8, $m5, $m15, 0
	shl $m8, $m7, 5
	add $m8, $m3, $m8
	st32 $m8, $m5, $m15, 1
	setzi $m0, .Lgemm_amp_worker
	runall $m0, $m5, 0
	sync SYNC_COMPUTE_SET
	add $m7, $m7, 1
	setzi $m8, INNER_MICRO_DIMENSION
	cmpult $m8, $m7, $m8
	brnz $m8, .Linner_group_\@
	add $m6, $m6, 1
	setzi $m8, BLOCK_DIMENSION / COLUMN_MICRO_DIMENSION
	cmpult $m8, $m6, $m8
	brnz $m8, .Lcolumn_group_\@
	br $m10
	.size \name, .-\name
	.endm

	GEMM_SUPERVISOR ipu_stack_gemm_f32_64_init, 1
	GEMM_SUPERVISOR ipu_stack_gemm_f32_64_accumulate, 0

	.macro AMP_WARM phase, c_offset
	ld64 $a0:1, $m2, $m15, \c_offset
	f32sisov2amp $azeros, $azero, $a0:1, \phase
	.endm

	.macro AMP_FIRST phase, a_offset, c_offset
	ld32 $a2, $m3, $m15, \a_offset
	ld64 $a0:1, $m2, $m15, (\c_offset + 32)
	f32sisov2amp $azeros, $a2, $a0:1, \phase
	.endm

	.macro AMP_MIDDLE phase, a_offset, c_offset
	ld32 $a2, $m3, $m15, \a_offset
	ld64 $a0:1, $m2, $m15, (\c_offset + 32)
	f32sisov2amp $a4:5, $a2, $a0:1, \phase
	st64 $a4:5, $m8, $m15, \c_offset
	.endm

	.macro AMP_LAST phase, a_offset, c_offset
	ld32 $a2, $m3, $m15, \a_offset
	f32sisov2amp $a4:5, $a2, $azeros, \phase
	st64 $a4:5, $m8, $m15, \c_offset
	.endm

	.macro AMP_DRAIN phase, c_offset
	f32sisov2amp $a4:5, $azero, $azeros, \phase
	st64 $a4:5, $m2, $m15, \c_offset
	.endm

	.worker
	.p2align 3
.Lgemm_amp_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	setzi $m5, BLOCK_DIMENSION / 6
	cmpult $m6, $m4, BLOCK_DIMENSION % 6
	add $m5, $m5, $m6
	mul $m6, $m4, BLOCK_DIMENSION / 6
	min $m7, $m4, BLOCK_DIMENSION % 6
	add $m6, $m6, $m7
	shl $m6, $m6, 8
	add $m2, $m2, $m6
	add $m3, $m3, $m6
	setzi $a0, CLEAR_AACC
	uput $FP_CLR, $a0

	AMP_WARM TAMP_F32_E4_P0, 0
	AMP_WARM TAMP_F32_E4_P1, 4
	AMP_WARM TAMP_F32_E4_P2, 1
	AMP_WARM TAMP_F32_E4_P3, 5
	AMP_WARM TAMP_F32_E4_P4, 2
	AMP_WARM TAMP_F32_E4_P5, 6
	AMP_WARM TAMP_F32_E4_P6, 3
	AMP_WARM TAMP_F32_E4_P7, 7

	AMP_FIRST TAMP_F32_E4_P0, 0, 0
	AMP_FIRST TAMP_F32_E4_P1, 1, 4
	AMP_FIRST TAMP_F32_E4_P2, 2, 1
	AMP_FIRST TAMP_F32_E4_P3, 3, 5
	AMP_FIRST TAMP_F32_E4_P4, 4, 2
	AMP_FIRST TAMP_F32_E4_P5, 5, 6
	AMP_FIRST TAMP_F32_E4_P6, 6, 3
	AMP_FIRST TAMP_F32_E4_P7, 7, 7
	add $m8, $m2, 0
	add $m2, $m2, BLOCK_DIMENSION * 4
	add $m3, $m3, BLOCK_DIMENSION * 4
	sub $m5, $m5, 1

.Lmiddle_row:
	cmpeq $m6, $m5, 1
	brnz $m6, .Llast_row
	AMP_MIDDLE TAMP_F32_E4_P0, 0, 0
	AMP_MIDDLE TAMP_F32_E4_P1, 1, 4
	AMP_MIDDLE TAMP_F32_E4_P2, 2, 1
	AMP_MIDDLE TAMP_F32_E4_P3, 3, 5
	AMP_MIDDLE TAMP_F32_E4_P4, 4, 2
	AMP_MIDDLE TAMP_F32_E4_P5, 5, 6
	AMP_MIDDLE TAMP_F32_E4_P6, 6, 3
	AMP_MIDDLE TAMP_F32_E4_P7, 7, 7
	add $m8, $m8, BLOCK_DIMENSION * 4
	add $m2, $m2, BLOCK_DIMENSION * 4
	add $m3, $m3, BLOCK_DIMENSION * 4
	sub $m5, $m5, 1
	bri .Lmiddle_row

.Llast_row:
	AMP_LAST TAMP_F32_E4_P0, 0, 0
	AMP_LAST TAMP_F32_E4_P1, 1, 4
	AMP_LAST TAMP_F32_E4_P2, 2, 1
	AMP_LAST TAMP_F32_E4_P3, 3, 5
	AMP_LAST TAMP_F32_E4_P4, 4, 2
	AMP_LAST TAMP_F32_E4_P5, 5, 6
	AMP_LAST TAMP_F32_E4_P6, 6, 3
	AMP_LAST TAMP_F32_E4_P7, 7, 7

	AMP_DRAIN TAMP_F32_E4_P0, 0
	AMP_DRAIN TAMP_F32_E4_P1, 4
	AMP_DRAIN TAMP_F32_E4_P2, 1
	AMP_DRAIN TAMP_F32_E4_P3, 5
	AMP_DRAIN TAMP_F32_E4_P4, 2
	AMP_DRAIN TAMP_F32_E4_P5, 6
	AMP_DRAIN TAMP_F32_E4_P6, 3
	AMP_DRAIN TAMP_F32_E4_P7, 7
	exitz $m15
