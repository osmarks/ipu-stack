#include "arch/gc_tile_defines.h"

#ifndef GEMM_INNER_BLOCK_DIMENSION
#define GEMM_INNER_BLOCK_DIMENSION 64
#endif
#ifndef GEMM_SMALL_ROWS
#define GEMM_SMALL_ROWS 12
#endif
#ifndef GEMM_LARGE_ROWS
#define GEMM_LARGE_ROWS GEMM_SMALL_ROWS
#endif
#ifndef GEMM_OUTPUT_COLUMNS
#define GEMM_OUTPUT_COLUMNS 64
#endif
#ifndef GEMM_RIGHT_BYTE_OFFSET
#define GEMM_RIGHT_BYTE_OFFSET 0
#endif
#ifndef GEMM_STOCHASTIC_ROUNDING
#define GEMM_STOCHASTIC_ROUNDING 1
#endif
#ifdef GEMM_NATIVE_FP8
#define GEMM_AMP f8v8hihov4amp
#define GEMM_TAMP_P0 TAMP_F8V8_E4_P0
#define GEMM_TAMP_P1 TAMP_F8V8_E4_P1
#define GEMM_TAMP_P2 TAMP_F8V8_E4_P2
#define GEMM_TAMP_P3 TAMP_F8V8_E4_P3
#define GEMM_INPUT_ELEMENT_BYTES 1
#define GEMM_INNER_MICRO_DIMENSION 32
#define GEMM_INPUT_FORMAT 3
#else
#define GEMM_AMP f16v4hihov4amp
#define GEMM_TAMP_P0 TAMP_F16V4_E4_P0
#define GEMM_TAMP_P1 TAMP_F16V4_E4_P1
#define GEMM_TAMP_P2 TAMP_F16V4_E4_P2
#define GEMM_TAMP_P3 TAMP_F16V4_E4_P3
#define GEMM_INPUT_ELEMENT_BYTES 2
#define GEMM_INNER_MICRO_DIMENSION 16
#define GEMM_INPUT_FORMAT 0
#endif
// The SDK's IPU21 assembly header omits this adjacent supervisor CSR name.
#define GEMM_FP_ISCL_INDEX (CSR_S_FP_INFMT__INDEX + 2)
#ifndef GEMM_INIT_SMALL_SYMBOL
#define GEMM_INIT_SMALL_SYMBOL ipu_stack_gemm_f16_init_small_rows
#endif
#ifndef GEMM_INIT_LARGE_SYMBOL
#define GEMM_INIT_LARGE_SYMBOL ipu_stack_gemm_f16_init_large_rows
#endif
#ifndef GEMM_ACCUMULATE_SMALL_SYMBOL
#define GEMM_ACCUMULATE_SMALL_SYMBOL ipu_stack_gemm_f16_accumulate_small_rows
#endif
#ifndef GEMM_ACCUMULATE_LARGE_SYMBOL
#define GEMM_ACCUMULATE_LARGE_SYMBOL ipu_stack_gemm_f16_accumulate_large_rows
#endif

	.text
	.allow_optimizations

	.set OUTPUT_COLUMNS, GEMM_OUTPUT_COLUMNS
	.set INNER_MICRO_DIMENSION, GEMM_INNER_MICRO_DIMENSION
	.set INNER_GROUPS, GEMM_INNER_BLOCK_DIMENSION / INNER_MICRO_DIMENSION
	.set COLUMN_MICRO_DIMENSION, 16
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set CLEAR_CWEI, 1 << 2
	.set CLEAR_AACC, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
	.set ENABLE_STOCHASTIC_ROUNDING, 1 << CSR_S_FP_ICTL__ESR__SHIFT

	.macro PARTITION_ROWS count
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	setzi $m5, (\count) / 6
	mul $m6, $m4, (\count) / 6
	.if (\count) % 6
	setzi $m9, (\count) % 6
	cmpult $m7, $m4, $m9
	add $m5, $m5, $m7
	min $m7, $m4, $m9
	add $m6, $m6, $m7
	.endif
	.endm

	.macro LOAD_AMP16_WEIGHTS
#ifdef GEMM_INTERLEAVED_WEIGHTS
	ld128putcs 0
	ld128putcs 2
	ld128putcs 4
	ld128putcs 6
	ld128putcs 32
	ld128putcs 34
	ld128putcs 36
	ld128putcs 38
	ld128putcs 8
	ld128putcs 10
	ld128putcs 12
	ld128putcs 14
	ld128putcs 40
	ld128putcs 42
	ld128putcs 44
	ld128putcs 46
	ld128putcs 16
	ld128putcs 18
	ld128putcs 20
	ld128putcs 22
	ld128putcs 48
	ld128putcs 50
	ld128putcs 52
	ld128putcs 54
	ld128putcs 24
	ld128putcs 26
	ld128putcs 28
	ld128putcs 30
	ld128putcs 56
	ld128putcs 58
	ld128putcs 60
	ld128putcs 62
#else
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
#endif
	.endm

	.macro GEMM_SUPERVISOR name, initial, rows, worker, retained_output_worker, retained_inner_worker
	.section .text.\name,"ax",@progbits
	.globl \name
	.p2align 2
	.type \name,@function
\name:
	.supervisor
	add $m11, $m11, -32
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	setzi $m8, \initial
	st32 $m8, $m11, $m15, 5
	st32 $m8, $m11, $m15, 7
	get $m8, $FP_ICTL
	st32 $m8, $m11, $m15, 6
#if GEMM_STOCHASTIC_ROUNDING
	setzi $m0, ENABLE_STOCHASTIC_ROUNDING
	or $m8, $m8, $m0
	put $FP_ICTL, $m8
#endif
	setzi $m0, GEMM_INPUT_FORMAT
	put CSR_S_FP_INFMT__INDEX, $m0
#ifdef GEMM_NATIVE_FP8
	put GEMM_FP_ISCL_INDEX, $m5
#else
	zero $m0
	put GEMM_FP_ISCL_INDEX, $m0
#endif
	setzi $m8, CLEAR_CWEI
	put $CR, $m8
	add $m4, $m4, GEMM_RIGHT_BYTE_OFFSET
	put $CCCSLOAD, $m4
	sync SYNC_COMPUTE_SET
	zero $m6
.Lcolumn_group_\@:
	zero $m7
.Linner_group_\@:
	LOAD_AMP16_WEIGHTS
	setzi $m9, \rows
	mul $m8, $m9, COLUMN_MICRO_DIMENSION * 2
	mul $m8, $m8, $m6
	add $m8, $m2, $m8
	st32 $m8, $m11, $m15, 0
	mul $m8, $m9, INNER_MICRO_DIMENSION * GEMM_INPUT_ELEMENT_BYTES
	mul $m8, $m8, $m7
	add $m8, $m3, $m8
	st32 $m8, $m11, $m15, 1
	zero $m0
	ld32 $m8, $m11, $m15, 7
	brz $m8, .Lstore_initial_\@
	brnz $m7, .Lstore_initial_\@
	setzi $m0, 1
.Lstore_initial_\@:
	st32 $m0, $m11, $m15, 5
	setzi $m0, \worker
	or $m8, $m6, $m7
	brz $m8, .Lrun_workers_\@
	setzi $m0, \retained_output_worker
	brz $m7, .Lrun_workers_\@
	setzi $m0, \retained_inner_worker
.Lrun_workers_\@:
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m7, $m7, 1
	setzi $m8, INNER_GROUPS
	cmpult $m8, $m7, $m8
	brnz $m8, .Linner_group_\@
	add $m6, $m6, 1
	setzi $m8, OUTPUT_COLUMNS / COLUMN_MICRO_DIMENSION
	cmpult $m8, $m6, $m8
	brnz $m8, .Lcolumn_group_\@
	ld32 $m8, $m11, $m15, 6
	put $FP_ICTL, $m8
	add $m11, $m11, 32
	br $m10
	.size \name, .-\name
	.endm

#ifdef GEMM_SINGLE_ROWS
	GEMM_SUPERVISOR GEMM_INIT_SMALL_SYMBOL, 1, GEMM_SMALL_ROWS, .Lgemm_f16_amp_worker_small, .Lgemm_f16_amp_worker_small_retained_output, .Lgemm_f16_amp_worker_small_retained_inner
	GEMM_SUPERVISOR GEMM_ACCUMULATE_SMALL_SYMBOL, 0, GEMM_SMALL_ROWS, .Lgemm_f16_amp_worker_small, .Lgemm_f16_amp_worker_small_retained_output, .Lgemm_f16_amp_worker_small_retained_inner
#else
	GEMM_SUPERVISOR GEMM_INIT_SMALL_SYMBOL, 1, GEMM_SMALL_ROWS, .Lgemm_f16_amp_worker_small, .Lgemm_f16_amp_worker_small_retained_output, .Lgemm_f16_amp_worker_small_retained_inner
	GEMM_SUPERVISOR GEMM_INIT_LARGE_SYMBOL, 1, GEMM_LARGE_ROWS, .Lgemm_f16_amp_worker_large, .Lgemm_f16_amp_worker_large_retained_output, .Lgemm_f16_amp_worker_large_retained_inner
	GEMM_SUPERVISOR GEMM_ACCUMULATE_SMALL_SYMBOL, 0, GEMM_SMALL_ROWS, .Lgemm_f16_amp_worker_small, .Lgemm_f16_amp_worker_small_retained_output, .Lgemm_f16_amp_worker_small_retained_inner
	GEMM_SUPERVISOR GEMM_ACCUMULATE_LARGE_SYMBOL, 0, GEMM_LARGE_ROWS, .Lgemm_f16_amp_worker_large, .Lgemm_f16_amp_worker_large_retained_output, .Lgemm_f16_amp_worker_large_retained_inner
#endif

	.worker
	.macro GEMM_WORKER name, retained_output_name, retained_inner_name, rows
	.p2align 3
\name:
	PARTITION_ROWS \rows
	shl $m6, $m6, 5
\retained_output_name:
	brz $m5, .Lgemm_f16_no_rows_\@
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	add $m9, $m0, $m6
	add $m2, $m1, $m6
	add $m3, $m9, 32
	add $m8, $m5, -1
	bri .Lgemm_f16_load_mode_\@
\retained_inner_name:
	brz $m5, .Lgemm_f16_no_rows_\@
	ld32 $m1, $mvertex_base, $m15, 1
	add $m2, $m1, $m6
.Lgemm_f16_load_mode_\@:
	ld32 $m7, $mvertex_base, $m15, 5
	brz $m7, .Lf16_accumulate_\@

	setzi $a6, CLEAR_AACC
	uput $FP_CLR, $a6
	ld64step $a0:1, $mzero, $m2+=, 1
	{
		ld64step $a0:1, $mzero, $m2+=, 1
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P0
	}
	{
		ld64step $a0:1, $mzero, $m2+=, 1
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P1
	}
	{
		ld64step $a0:1, $mzero, $m2+=, 1
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P2
	}
	{
		ld64step $a0:1, $mzero, $m2+=, 1
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P3
	}
	{
		ld64step $a0:1, $mzero, $m2+=, 1
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P0
	}
	tapack $m10:11, $m2, $m3, $m9
	{
		rpt $m8, 3
		fnop
	}
	{
		ldst64pace $a0:1, $a6:7, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P1
	}
	{
		ldst64pace $a0:1, $a6:7, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P2
	}
	{
		ldst64pace $a0:1, $a6:7, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P3
	}
	{
		ldst64pace $a0:1, $a6:7, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $azeros, GEMM_TAMP_P0
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		GEMM_AMP $a6:7, $azeros, $azeros, GEMM_TAMP_P1
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		GEMM_AMP $a6:7, $azeros, $azeros, GEMM_TAMP_P2
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		GEMM_AMP $a6:7, $azeros, $azeros, GEMM_TAMP_P3
	}
	st64pace $a6:7, $m10:11+=, $mzero, 0b00
	exitz $mzero

.Lf16_accumulate_\@:
	setzi $a6, CLEAR_AACC
	uput $FP_CLR, $a6
	ld128 $a0:3, $m9, $m15, 0
	{
		add $m2, $m2, 0
		GEMM_AMP $a6:7, $azeros, $a0:1, GEMM_TAMP_P0
	}
	{
		ld128 $a0:3, $m9, $m15, 1
		GEMM_AMP $a6:7, $azeros, $a2:3, GEMM_TAMP_P1
	}
	{
		tapack $m10:11, $m2, $m3, $m9
		GEMM_AMP $a6:7, $azeros, $a0:1, GEMM_TAMP_P2
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $azeros, $a2:3, GEMM_TAMP_P3
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P0
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P1
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P2
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P3
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P0
	}
	{
		ld2x64pace $a0:1, $a2:3, $m10:11+=, $mzero, 0b0000
		GEMM_AMP $a4:5, $a0:1, $a2:3, GEMM_TAMP_P1
	}
	rpt $m8, 3
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P2
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b000000
		GEMM_AMP $a4:5, $a0:1, $a2:3, GEMM_TAMP_P3
	}
	{
		ld2xst64pace $a0:3, $a6:7, $m10:11+=, $mzero, 0b000000
		GEMM_AMP $a6:7, $a0:1, $a2:3, GEMM_TAMP_P0
	}
	{
		ld2xst64pace $a0:3, $a4:5, $m10:11+=, $mzero, 0b000000
		GEMM_AMP $a4:5, $a0:1, $a2:3, GEMM_TAMP_P1
	}
	{
		st64pace $a6:7, $m10:11+=, $mzero, 0b00
		GEMM_AMP $a6:7, $azeros, $azeros, GEMM_TAMP_P2
	}
	{
		st64pace $a4:5, $m10:11+=, $mzero, 0b00
		GEMM_AMP $a4:5, $azeros, $a2:3, GEMM_TAMP_P3
	}
	st64pace $a6:7, $m10:11+=, $mzero, 0b00
	st64pace $a4:5, $m10:11+=, $mzero, 0b00
	exitz $mzero
.Lgemm_f16_no_rows_\@:
	exitz $mzero
	.endm

	GEMM_WORKER .Lgemm_f16_amp_worker_small, .Lgemm_f16_amp_worker_small_retained_output, .Lgemm_f16_amp_worker_small_retained_inner, GEMM_SMALL_ROWS
#ifndef GEMM_SINGLE_ROWS
	GEMM_WORKER .Lgemm_f16_amp_worker_large, .Lgemm_f16_amp_worker_large_retained_output, .Lgemm_f16_amp_worker_large_retained_inner, GEMM_LARGE_ROWS
#endif
