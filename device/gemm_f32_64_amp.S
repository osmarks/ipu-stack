#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set BLOCK_DIMENSION, 64
	.set INNER_MICRO_DIMENSION, 8
	.set COLUMN_MICRO_DIMENSION, 16
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	// The SDK's supervisor CR header does not expose this shift to assembly.
	.set CLEAR_CWEI, 1 << 2
	.set CLEAR_AACC, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
	// Packed increments for contiguous 8-float A and 16-float C rows.
	.set PACE_ROW_STRIDE, (1 << 10) | 1

	.macro PARTITION_64BIT_UNITS units
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	setzi $m5, (\units) / 6
	cmpult $m6, $m4, (\units) % 6
	add $m5, $m5, $m6
	mul $m6, $m4, (\units) / 6
	min $m7, $m4, (\units) % 6
	add $m6, $m6, $m7
	shl $m6, $m6, 3
	.endm

	.section .text.ipu_stack_copy_4096_u32,"ax",@progbits
	.globl ipu_stack_copy_4096_u32
	.p2align 2
	.type ipu_stack_copy_4096_u32,@function
ipu_stack_copy_4096_u32:
	.supervisor
	add $m11, $m11, -8
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	setzi $m0, .Lcopy_4096_u32_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 8
	br $m10
	.size ipu_stack_copy_4096_u32, .-ipu_stack_copy_4096_u32

	.worker
	.p2align 2
.Lcopy_4096_u32_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	PARTITION_64BIT_UNITS BLOCK_DIMENSION * BLOCK_DIMENSION / 2
	add $m2, $m2, $m6
	add $m3, $m3, $m6
.Lcopy_64:
	ld64 $a0:1, $m3, $m15, 0
	st64 $a0:1, $m2, $m15, 0
	add $m3, $m3, 8
	add $m2, $m2, 8
	sub $m5, $m5, 1
	brnz $m5, .Lcopy_64
	exitz $mzero

	.worker
	.p2align 2
.Lzero_4096_u32_worker:
	mov $m2, $mvertex_base
	PARTITION_64BIT_UNITS BLOCK_DIMENSION * BLOCK_DIMENSION / 2
	add $m2, $m2, $m6
.Lzero_64:
	st64 $azeros, $m2, $m15, 0
	add $m2, $m2, 8
	sub $m5, $m5, 1
	brnz $m5, .Lzero_64
	exitz $mzero

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
	setzi $m0, .Lzero_4096_u32_worker
	runall $m0, $m2, 0
	sync SYNC_COMPUTE_SET
	.endif
	add $m11, $m11, -8
	setzi $m5, CLEAR_CWEI
	put $CR, $m5
	put $CCCSLOAD, $m4
	sync SYNC_COMPUTE_SET
	zero $m6
.Lcolumn_group_\@:
	zero $m7
.Linner_group_\@:
	LOAD_AMP16_WEIGHTS
	mov $m5, $m11
	shl $m8, $m6, 12
	add $m8, $m2, $m8
	st32 $m8, $m5, $m15, 0
	shl $m8, $m7, 11
	add $m8, $m3, $m8
	st32 $m8, $m5, $m15, 1
	setzi $m0, .Lgemm_amp_pace_worker
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
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 8
	br $m10
	.size \name, .-\name
	.endm

	GEMM_SUPERVISOR ipu_stack_gemm_f32_64_init, 1
	GEMM_SUPERVISOR ipu_stack_gemm_f32_64_accumulate, 0

	.worker
	.p2align 3
.Lgemm_amp_pace_worker:
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
	shl $m7, $m6, 6
	shl $m6, $m6, 5
	add $m2, $m2, $m7
	add $m3, $m3, $m6
	add $m5, $m5, -1
	add $m4, $m5, -1
	setzi $m7, PACE_ROW_STRIDE
	setzi $a7, CLEAR_AACC
	uput $FP_CLR, $a7
	add $m9, $m2, COLUMN_MICRO_DIMENSION * 4

	ld128 $a0:3, $m2, 0
	f32sisov2amp $a4:5, $azero, $a0:1, TAMP_F32_E4_P0
	{
		ld128 $a0:3, $m2, 1
		f32sisov2amp $a4:5, $azero, $a2:3, TAMP_F32_E4_P1
	}
	f32sisov2amp $a4:5, $azero, $a0:1, TAMP_F32_E4_P2
	{
		ld128 $a0:3, $m2, 2
		f32sisov2amp $a4:5, $azero, $a2:3, TAMP_F32_E4_P3
	}
	f32sisov2amp $a4:5, $azero, $a0:1, TAMP_F32_E4_P4
	{
		ld128 $a0:3, $m2, 3
		f32sisov2amp $a4:5, $azero, $a2:3, TAMP_F32_E4_P5
	}
	{
		tapack $m10:11, $m3, $m9, $m2
		f32sisov2amp $a4:5, $azero, $a0:1, TAMP_F32_E4_P6
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $m7, 0b1000
		f32sisov2amp $a4:5, $azero, $a2:3, TAMP_F32_E4_P7
	}
	{
		ld2x64pace $azeros, $a2:3, $m10:11+=, $m7, 0b1011
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P0
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $m7, 0b1000
		f32sisov2amp $a4:5, $a1, $a2:3, TAMP_F32_E4_P1
	}
	{
		ld2x64pace $azeros, $a2:3, $m10:11+=, $m7, 0b1011
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P2
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $m7, 0b1000
		f32sisov2amp $a4:5, $a1, $a2:3, TAMP_F32_E4_P3
	}
	{
		ld2x64pace $azeros, $a2:3, $m10:11+=, $m7, 0b1011
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P4
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $m7, 0b1001
		f32sisov2amp $a4:5, $a1, $a2:3, TAMP_F32_E4_P5
	}
	{
		ld2x64pace $azeros, $a2:3, $m10:11+=, $m7, 0b1011
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P6
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $m7, 0b1010
		f32sisov2amp $a4:5, $a1, $a2:3, TAMP_F32_E4_P7
	}
	{
		ld2x64pace $azeros, $a2:3, $m10:11+=, $m7, 0b1011
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P0
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $m7, 0b1011
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P1
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b0000
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P2
	}
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000001
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P3
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b000000
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P4
	}

	{
		rpt $m4, (.Lpace_loop_end - .Lpace_loop_start) / 8 - 1
		fnop
	}
.Lpace_loop_start:
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000001
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P5
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $m7, 0b000001
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P6
	}
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000001
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P7
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b000000
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P0
	}
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000001
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P1
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b000000
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P2
	}
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000001
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P3
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b000000
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P4
	}
.Lpace_loop_end:
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000001
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P5
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b001111
		f32sisov2amp $a4:5, $a0, $a2:3, TAMP_F32_E4_P6
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a6:7, $a1, $a2:3, TAMP_F32_E4_P7
	}
	{
		st64pace $a4:5, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a4:5, $azero, $azeros, TAMP_F32_E4_P0
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a6:7, $azero, $azeros, TAMP_F32_E4_P1
	}
	{
		st64pace $a4:5, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a4:5, $azero, $azeros, TAMP_F32_E4_P2
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a6:7, $azero, $azeros, TAMP_F32_E4_P3
	}
	{
		st64pace $a4:5, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a4:5, $azero, $azeros, TAMP_F32_E4_P4
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a6:7, $azero, $azeros, TAMP_F32_E4_P5
	}
	{
		st64pace $a4:5, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a4:5, $azero, $azeros, TAMP_F32_E4_P6
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		f32sisov2amp $a6:7, $azero, $azeros, TAMP_F32_E4_P7
	}
	st64pace $a4:5, $m10:11+=, $mzero, 0b00
	st64pace $a6:7, $m10:11+=, $mzero, 0b00
	exitz $mzero
