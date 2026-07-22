#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set TILE_MUX_HOST, 0x600
	.set TILE_MUX_EXCHANGE, 0x640
	.set WORKER_BASE_PLACEHOLDER, 0
	.set WORKER_BASE_STRIDE, 0x100
	.set HOST_RUN_DESCRIPTOR_BYTES, 3 * 4
	.set HOST_PACKET_ADDRESS, 0x50000
	.set SYNC_ALL, 7
	.set SYNC_COMPUTE_SET, 2
	.set SYNC_SUPERVISOR, 3
	.set OPCODE_SYNC, 0x41800000
	.set PROGRAM_ADDRESS_PLACEHOLDER, 0
	.set COMPLETION_ADDRESS_PLACEHOLDER, 0
	.set PRNG_SEED_BASE_PLACEHOLDER, 0
	.set HOST_STAGING_ADDRESS_PLACEHOLDER, 0
	.set PRNG_WARMUP_REPEAT_COUNT, 15

	.section .text.ipu_stack_static_start,"ax",@progbits
	.globl ipu_stack_static_start
	.p2align 2
	nop
	.type ipu_stack_static_start,@function
ipu_stack_static_start:
	.supervisor
	.long OPCODE_SYNC | SYNC_SUPERVISOR
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	put 0xa7, $m15
	.globl ipu_stack_static_worker_base
ipu_stack_static_worker_base:
	setzi $m11, WORKER_BASE_PLACEHOLDER
	put 0x21, $m11
	add $m11, $m11, WORKER_BASE_STRIDE
	put 0x22, $m11
	add $m11, $m11, WORKER_BASE_STRIDE
	put 0x23, $m11
	add $m11, $m11, WORKER_BASE_STRIDE
	put 0x24, $m11
	add $m11, $m11, WORKER_BASE_STRIDE
	put 0x25, $m11
	add $m11, $m11, WORKER_BASE_STRIDE
	put 0x26, $m11
	add $m11, $m11, WORKER_BASE_STRIDE
	setzi $m0, ipu_stack_static_worker_init
	runall $m0, $m15, 0
	setzi $m0, TILE_MUX_HOST
	put 0xa0, $m0
	.long OPCODE_SYNC | SYNC_ALL
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	.globl ipu_stack_static_program_address
ipu_stack_static_program_address:
	setzi $m0, PROGRAM_ADDRESS_PLACEHOLDER
	br $m0
	.size ipu_stack_static_start, .-ipu_stack_static_start

	.section .text.ipu_stack_static_worker_init,"ax",@progbits
	.globl ipu_stack_static_worker_init
	.p2align 2
	.type ipu_stack_static_worker_init,@function
ipu_stack_static_worker_init:
	.worker
	.globl ipu_stack_static_worker_sync_context_base
ipu_stack_static_worker_sync_context_base:
	setzi $m0, 0
	get $m1, 1
	and $m1, $m1, 7
	or $m0, $m0, $m1
	st32 $m0, $m12, $m15, 0
	ld32 $a0, $m12, $m15, 0
	put 0x7, $a0
	.globl ipu_stack_static_prng_seed_base
ipu_stack_static_prng_seed_base:
	setzi $m0, PRNG_SEED_BASE_PLACEHOLDER
	or $m0, $m0, $m1
	st32 $m0, $m12, $m15, 0
	ld32 $a0, $m12, $m15, 0
	uput $PRNG_SEED, $a0
	{ rpt PRNG_WARMUP_REPEAT_COUNT, 0
	  fnop }
	{ nop
	  urand64 $azeros }
	exitz $m15
	.size ipu_stack_static_worker_init, .-ipu_stack_static_worker_init

	.section .text.ipu_stack_static_worker_barrier,"ax",@progbits
	.globl ipu_stack_static_worker_barrier
	.p2align 2
	.type ipu_stack_static_worker_barrier,@function
ipu_stack_static_worker_barrier:
	.supervisor
	setzi $m0, ipu_stack_static_worker_barrier_worker
	runall $m0, $m15, 0
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	br $m7
	.size ipu_stack_static_worker_barrier, .-ipu_stack_static_worker_barrier

	.section .text.ipu_stack_static_worker_barrier_worker,"ax",@progbits
	.p2align 2
	.type ipu_stack_static_worker_barrier_worker,@function
ipu_stack_static_worker_barrier_worker:
	.worker
	exitz $m15
	.size ipu_stack_static_worker_barrier_worker, .-ipu_stack_static_worker_barrier_worker

	.section .text.ipu_stack_static_copy_u32,"ax",@progbits
	.globl ipu_stack_static_copy_u32
	.p2align 2
	.type ipu_stack_static_copy_u32,@function
ipu_stack_static_copy_u32:
	.supervisor
.Lcopy_u32:
	ld32 $m0, $m3, $m15, 0
	st32 $m0, $m2, $m15, 0
	add $m3, $m3, 4
	add $m2, $m2, 4
	sub $m4, $m4, 1
	brnz $m4, .Lcopy_u32
	br $m10
	.size ipu_stack_static_copy_u32, .-ipu_stack_static_copy_u32

	.section .text.ipu_stack_copy_u64,"ax",@progbits
	.globl ipu_stack_copy_u64
	.p2align 2
	.type ipu_stack_copy_u64,@function
ipu_stack_copy_u64:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m6, $m11, $m15, 2
	st32 $m7, $m11, $m15, 3
	setzi $m0, .Lcopy_u64_worker
	runall $m0, $m11, 0
	sync SYNC_COMPUTE_SET
	add $m11, $m11, 16
	br $m10
	.size ipu_stack_copy_u64, .-ipu_stack_copy_u64

	.worker
	.p2align 2
.Lcopy_u64_worker:
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	ld32 $m5, $mvertex_base, $m15, 2
	ld32 $m6, $mvertex_base, $m15, 3
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
	cmpult $m7, $m4, $m6
	mul $m8, $m4, $m5
	min $m9, $m4, $m6
	add $m8, $m8, $m9
	shl $m8, $m8, 3
	add $m2, $m2, $m8
	add $m3, $m3, $m8
	add $m5, $m5, $m7
.Lcopy_u64_worker_loop:
	ld64 $a0:1, $m3, $m15, 0
	st64 $a0:1, $m2, $m15, 0
	add $m3, $m3, 8
	add $m2, $m2, 8
	sub $m5, $m5, 1
	brnz $m5, .Lcopy_u64_worker_loop
	exitz $m15

	.section .text.ipu_stack_static_repeat_call,"ax",@progbits
	.globl ipu_stack_static_repeat_call
	.p2align 2
	.type ipu_stack_static_repeat_call,@function
ipu_stack_static_repeat_call:
	.supervisor
.Lrepeat_call:
	setzi $m10, .Lrepeat_return
	br $m3
.Lrepeat_return:
	sub $m2, $m2, 1
	brnz $m2, .Lrepeat_call
	br $m9
	.size ipu_stack_static_repeat_call, .-ipu_stack_static_repeat_call

	.section .text.ipu_stack_static_exchange_compute_run,"ax",@progbits
	.globl ipu_stack_static_exchange_compute_run
	.p2align 2
	.type ipu_stack_static_exchange_compute_run,@function
ipu_stack_static_exchange_compute_run:
	.supervisor
	add $m11, $m11, -32
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
	st32 $m6, $m11, $m15, 4
	st32 $m7, $m11, $m15, 5
	st32 $m9, $m11, $m15, 6
.Lexchange_compute_next:
	ld32 $m8, $m11, $m15, 1
	ld32 $m8, $m8, $m15, 0
	.long OPCODE_SYNC | SYNC_SUPERVISOR
	and $m0, $m8, 1
	brz $m0, .Lexchange_compute_no_worker_barrier
	setzi $m7, .Lexchange_compute_after_worker_barrier
	setzi $m1, ipu_stack_static_worker_barrier
	br $m1
.Lexchange_compute_after_worker_barrier:
	ld32 $m8, $m11, $m15, 1
	ld32 $m8, $m8, $m15, 0
	setzi $m0, 1
.Lexchange_compute_no_worker_barrier:
	sub $m8, $m8, $m0
	setzi $m10, .Lexchange_compute_after_exchange
	br $m8
.Lexchange_compute_after_exchange:
	ld32 $m2, $m11, $m15, 2
	ld32 $m3, $m11, $m15, 3
	ld32 $m4, $m11, $m15, 4
	ld32 $m0, $m11, $m15, 5
	setzi $m10, .Lexchange_compute_after_kernel
	br $m0
.Lexchange_compute_after_kernel:
	ld32 $m3, $m11, $m15, 1
	add $m3, $m3, 4
	st32 $m3, $m11, $m15, 1
	ld32 $m2, $m11, $m15, 0
	sub $m2, $m2, 1
	st32 $m2, $m11, $m15, 0
	brnz $m2, .Lexchange_compute_next
	ld32 $m9, $m11, $m15, 6
	add $m11, $m11, 32
	br $m9
	.size ipu_stack_static_exchange_compute_run, .-ipu_stack_static_exchange_compute_run

	.section .text.ipu_stack_static_template_patch,"ax",@progbits
	.globl ipu_stack_static_template_patch
	.p2align 2
	.type ipu_stack_static_template_patch,@function
ipu_stack_static_template_patch:
	.supervisor
	add $m11, $m11, -32
	brz $m7, .Ltemplate_patch_primary_segment
	ld32 $m4, $m11, $m15, 9
	setzi $m0, .Ltemplate_patch_segment_ready
	br $m0
.Ltemplate_patch_primary_segment:
	ld32 $m4, $m11, $m15, 8
.Ltemplate_patch_segment_ready:
	ld32 $m0, $m3, $m15, 0
	setzi $m10, 0xffff
	and $m10, $m0, $m10
	shl $m10, $m10, 2
	add $m4, $m4, $m10
	shr $m2, $m0, 16
	add $m3, $m3, 4
	st32 $m4, $m11, $m15, 0
	st32 $m9, $m11, $m15, 3
	st32 $m3, $m11, $m15, 5
	add $m10, $m3, 0
	add $m6, $m2, 0
	zero $m1
.Ltemplate_patch_changed_group:
	ld32 $m4, $m10, $m15, 0
.Ltemplate_patch_changed_bit:
	and $m0, $m4, 1
	add $m1, $m1, $m0
	shr $m4, $m4, 1
	brnz $m4, .Ltemplate_patch_changed_bit
	add $m10, $m10, 4
	sub $m6, $m6, 1
	brnz $m6, .Ltemplate_patch_changed_group
	st32 $m10, $m11, $m15, 1
	add $m6, $m1, 31
	shr $m6, $m6, 5
	zero $m7
.Ltemplate_patch_type_group:
	ld32 $m4, $m10, $m15, 0
.Ltemplate_patch_type_bit:
	and $m0, $m4, 1
	add $m7, $m7, $m0
	shr $m4, $m4, 1
	brnz $m4, .Ltemplate_patch_type_bit
	add $m10, $m10, 4
	sub $m6, $m6, 1
	brnz $m6, .Ltemplate_patch_type_group
	add $m3, $m10, 0
	add $m5, $m7, 1
	shr $m5, $m5, 1
	shl $m5, $m5, 2
	add $m5, $m3, $m5
	shl $m2, $m2, 5
	zero $m9
	zero $m7
	zero $m6
.Ltemplate_patch_mask:
	ld32 $m10, $m11, $m15, 5
	ld32 $m8, $m10, $m15, 0
	add $m10, $m10, 4
	st32 $m10, $m11, $m15, 5
	setzi $m1, 32
.Ltemplate_patch_bit:
	and $m0, $m8, 1
	brz $m0, .Ltemplate_patch_advance
	shl $m10, $m7, 2
	ld32 $m0, $m11, $m15, 0
	add $m10, $m0, $m10
	st32 $m10, $m11, $m15, 4
	brnz $m6, .Ltemplate_patch_have_type
	ld32 $m10, $m11, $m15, 1
	ld32 $m4, $m10, $m15, 0
	add $m10, $m10, 4
	st32 $m10, $m11, $m15, 1
	setzi $m6, 32
.Ltemplate_patch_have_type:
	and $m0, $m4, 1
	shr $m4, $m4, 1
	sub $m6, $m6, 1
	brz $m0, .Ltemplate_patch_wide
	brnz $m9, .Ltemplate_patch_narrow_high
	ld32 $m0, $m3, $m15, 0
	st32 $m0, $m11, $m15, 7
	shl $m0, $m0, 16
	shr $m0, $m0, 16
	setzi $m9, 1
	setzi $m10, .Ltemplate_patch_narrow_value
	br $m10
.Ltemplate_patch_narrow_high:
	ld32 $m0, $m11, $m15, 7
	shr $m0, $m0, 16
	zero $m9
	add $m3, $m3, 4
.Ltemplate_patch_narrow_value:
	setzi $m10, 0x8000
	xor $m0, $m0, $m10
	sub $m0, $m0, $m10
	ld32 $m10, $m11, $m15, 4
	ld32 $m10, $m10, $m15, 0
	add $m0, $m0, $m10
	setzi $m10, .Ltemplate_patch_store
	br $m10
.Ltemplate_patch_wide:
	ld32 $m0, $m5, $m15, 0
	add $m5, $m5, 4
.Ltemplate_patch_store:
	ld32 $m10, $m11, $m15, 4
	st32 $m0, $m10, $m15, 0
.Ltemplate_patch_advance:
	shr $m8, $m8, 1
	add $m7, $m7, 1
	sub $m2, $m2, 1
	brz $m2, .Ltemplate_patch_done
	sub $m1, $m1, 1
	brnz $m1, .Ltemplate_patch_bit
	setzi $m0, .Ltemplate_patch_mask
	br $m0
.Ltemplate_patch_done:
	ld32 $m9, $m11, $m15, 3
	add $m11, $m11, 32
	br $m9
	.size ipu_stack_static_template_patch, .-ipu_stack_static_template_patch


	.section .text.ipu_stack_static_host_run,"ax",@progbits
	.globl ipu_stack_static_host_run
	.p2align 2
	.type ipu_stack_static_host_run,@function
ipu_stack_static_host_run:
	.supervisor
	add $m11, $m11, -16
	st32 $m4, $m11, $m15, 2
.Lhost_run_next:
	ld32 $m0, $m3, $m15, 1
	shr $m5, $m0, 24
	brz $m5, .Lhost_run_call
	shr $m6, $m0, 20
	and $m6, $m6, 8
	setzi $m1, HOST_PACKET_ADDRESS
	add $m6, $m6, $m1
	ld32 $m7, $m3, $m15, 2
.Lhost_run_packet_copy:
	ld32 $m0, $m7, $m15, 0
	st32 $m0, $m6, $m15, 0
	add $m7, $m7, 4
	add $m6, $m6, 4
	sub $m5, $m5, 1
	brnz $m5, .Lhost_run_packet_copy
.Lhost_run_call:
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	ld32 $m0, $m11, $m15, 2
	setzi $m10, .Lhost_run_return
	br $m0
.Lhost_run_return:
	ld32 $m2, $m11, $m15, 0
	ld32 $m3, $m11, $m15, 1
	ld32 $m5, $m3, $m15, 1
	shl $m5, $m5, 9
	shr $m5, $m5, 9
	brz $m5, .Lhost_run_advance
	ld32 $m6, $m3, $m15, 0
	.globl ipu_stack_static_host_staging_address
ipu_stack_static_host_staging_address:
	setzi $m7, HOST_STAGING_ADDRESS_PLACEHOLDER
.Lhost_run_copy:
	ld32 $m0, $m7, $m15, 0
	st32 $m0, $m6, $m15, 0
	add $m7, $m7, 4
	add $m6, $m6, 4
	sub $m5, $m5, 1
	brnz $m5, .Lhost_run_copy
.Lhost_run_advance:
	add $m3, $m3, HOST_RUN_DESCRIPTOR_BYTES
	sub $m2, $m2, 1
	brnz $m2, .Lhost_run_next
	add $m11, $m11, 16
	br $m9
	.size ipu_stack_static_host_run, .-ipu_stack_static_host_run

	.section .text.ipu_stack_static_sample_cycle,"ax",@progbits
	.globl ipu_stack_static_sample_cycle
	.p2align 2
	.type ipu_stack_static_sample_cycle,@function
ipu_stack_static_sample_cycle:
	.supervisor
	.globl ipu_stack_static_sample_worker_base
ipu_stack_static_sample_worker_base:
	setzi $m5, WORKER_BASE_PLACEHOLDER
	st32 $m2, $m5, $m15, 0
	setzi $m0, ipu_stack_static_sample_cycle_worker
	runall $m0, $m5, 0
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	br $m10
	.size ipu_stack_static_sample_cycle, .-ipu_stack_static_sample_cycle

	.section .text.ipu_stack_static_sample_cycle_next,"ax",@progbits
	.globl ipu_stack_static_sample_cycle_next
	.p2align 2
	.type ipu_stack_static_sample_cycle_next,@function
ipu_stack_static_sample_cycle_next:
	.supervisor
	.globl ipu_stack_static_sample_next_worker_base
ipu_stack_static_sample_next_worker_base:
	setzi $m5, WORKER_BASE_PLACEHOLDER
	ld32 $m2, $m5, $m15, 0
	add $m2, $m2, 4
	st32 $m2, $m5, $m15, 0
	setzi $m0, ipu_stack_static_sample_cycle_worker
	runall $m0, $m5, 0
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	br $m10
	.size ipu_stack_static_sample_cycle_next, .-ipu_stack_static_sample_cycle_next

	.section .text.ipu_stack_static_sample_cycle_worker,"ax",@progbits
	.p2align 2
	.type ipu_stack_static_sample_cycle_worker,@function
ipu_stack_static_sample_cycle_worker:
	.worker
	get $m1, $WSR
	and $m1, $m1, 7
	brnz $m1, .Lsample_cycle_exit
	get $m0, $COUNT_L
	ld32 $m2, $mvertex_base, $m15, 0
	st32 $m0, $m2, $m15, 0
.Lsample_cycle_exit:
	exitz $m15
	.size ipu_stack_static_sample_cycle_worker, .-ipu_stack_static_sample_cycle_worker

	.section .text.ipu_stack_static_complete,"ax",@progbits
	.globl ipu_stack_static_complete
	.p2align 2
	.type ipu_stack_static_complete,@function
ipu_stack_static_complete:
	.supervisor
	setzi $m0, 1
	.globl ipu_stack_static_completion_address
ipu_stack_static_completion_address:
	setzi $m1, COMPLETION_ADDRESS_PLACEHOLDER
	st32 $m0, $m1, $m15, 0
	zero $m0
	br $m0
	.size ipu_stack_static_complete, .-ipu_stack_static_complete
