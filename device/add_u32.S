#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL

	.section .text.ipu_stack_add_u32,"ax",@progbits
	.globl ipu_stack_add_u32
	.p2align 2
	.type ipu_stack_add_u32,@function
ipu_stack_add_u32:
	.supervisor
	ld32 $m8, $m3, $m15, 0
	ld32 $m9, $m4, $m15, 0
	add $m8, $m8, $m9
	st32 $m8, $m2, $m15, 0
	br $m10
	.size ipu_stack_add_u32, .-ipu_stack_add_u32

	.section .text.ipu_stack_add,"ax",@progbits
	.globl ipu_stack_add
	.p2align 2
	.type ipu_stack_add,@function
ipu_stack_add:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	setzi $m0, .Ladd_f32_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size ipu_stack_add, .-ipu_stack_add

	.worker
	.p2align 3
.Ladd_f32_worker:
	get $m0, $WSR
	and $m0, $m0, CSR_W_WSR__CTXTID_M1__MASK
	brnz $m0, .Ladd_f32_done
	ld32 $m0, $mvertex_base, $m15, 0
	ld32 $m1, $mvertex_base, $m15, 1
	ld32 $m2, $mvertex_base, $m15, 2
	ld32 $a0, $m1, $m15, 0
	ld32 $a2, $m2, $m15, 0
	f32v2add $a4:5, $a0:1, $a2:3
	st32 $a4, $m0, $m15, 0
.Ladd_f32_done:
	exitz $m15
