#include "arch/gc_tile_defines.h"

	.text
	.allow_optimizations

	.set TILE_MUX_HOST, 0x600
	.set TILE_MUX_EXCHANGE, 0x640
	.set WORKER_BASE_STRIDE, 0x100
	.set HOST_RUN_DESCRIPTOR_BYTES, 3 * 4
	.set HOST_PACKET_ADDRESS, 0x50000
	.set SYNC_ALL, 7
	.set SYNC_COMPUTE_SET, TEXCH_SYNCZONE_LOCAL
	.set SYNC_SUPERVISOR, 3
	.set OPCODE_SYNC, 0x41800000
	.set PRNG_WARMUP_REPEAT_COUNT, 15

	.extern ipu_stack_static_worker_sync_context
	.extern ipu_stack_static_worker_stack_base
	.extern ipu_stack_static_prng_seed
	.extern ipu_stack_static_program
	.extern ipu_stack_static_host_staging
	.extern ipu_stack_static_completion

	.section .text.ipu_stack_static_start,"ax",@progbits
	.globl ipu_stack_static_start
	.p2align 2
	nop
	.type ipu_stack_static_start,@function
ipu_stack_static_start:
	.supervisor
	.long OPCODE_SYNC | SYNC_SUPERVISOR
	// The bootloader may reach application entry before every worker retires.
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	put 0xa7, $m15
	setzi $m11, ipu_stack_static_worker_stack_base
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
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	setzi $m0, TILE_MUX_HOST
	put 0xa0, $m0
	.long OPCODE_SYNC | SYNC_ALL
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	setzi $m0, ipu_stack_static_program
	br $m0
	.size ipu_stack_static_start, .-ipu_stack_static_start

	.section .text.ipu_stack_static_worker_init,"ax",@progbits
	.globl ipu_stack_static_worker_init
	.p2align 2
	.type ipu_stack_static_worker_init,@function
ipu_stack_static_worker_init:
	.worker
	setzi $m0, ipu_stack_static_worker_sync_context
	get $m1, 1
	and $m1, $m1, 7
	or $m0, $m0, $m1
	st32 $m0, $m12, $m15, 0
	ld32 $a0, $m12, $m15, 0
	put 0x7, $a0
	setzi $m0, ipu_stack_static_prng_seed
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
	// Runtime helper ABI: m2 destination, m3 source, m4 word count, m10 return.
	.globl ipu_stack_static_copy_u32
	.p2align 2
	.type ipu_stack_static_copy_u32,@function
ipu_stack_static_copy_u32:
	.supervisor
	brz $m4, .Lcopy_u32_done
.Lcopy_u32:
	ld32 $m0, $m3, $m15, 0
	st32 $m0, $m2, $m15, 0
	add $m3, $m3, 4
	add $m2, $m2, 4
	sub $m4, $m4, 1
	brnz $m4, .Lcopy_u32
.Lcopy_u32_done:
	br $m10
	.size ipu_stack_static_copy_u32, .-ipu_stack_static_copy_u32

	.section .text.ipu_stack_static_copy_u16,"ax",@progbits
	// Runtime helper ABI: m4 entry count and m10 points to inline
	// (source address, destination address) pairs. It returns past the table.
	.globl ipu_stack_static_copy_u16
	.p2align 2
	.type ipu_stack_static_copy_u16,@function
ipu_stack_static_copy_u16:
	.supervisor
	brz $m4, .Lcopy_u16_done
.Lcopy_u16:
	ld32 $m3, $m10, $m15, 0
	ld32 $m2, $m10, $m15, 1
	add $m10, $m10, 8
	mov $m6, $m3
	shr $m6, $m6, 2
	shl $m6, $m6, 2
	ld32 $m0, $m6, $m15, 0
	setzi $m5, 2
	and $m6, $m3, $m5
	brz $m6, .Lcopy_u16_source_low
	shr $m0, $m0, 16
.Lcopy_u16_source_low:
	setzi $m5, 65535
	and $m0, $m0, $m5
	mov $m6, $m2
	shr $m6, $m6, 2
	shl $m6, $m6, 2
	ld32 $m1, $m6, $m15, 0
	setzi $m5, 2
	and $m7, $m2, $m5
	brz $m7, .Lcopy_u16_destination_low
	shl $m0, $m0, 16
	setzi $m5, 65535
	and $m1, $m1, $m5
	bri .Lcopy_u16_store
.Lcopy_u16_destination_low:
	setzi $m5, 65535
	shl $m5, $m5, 16
	and $m1, $m1, $m5
.Lcopy_u16_store:
	or $m0, $m0, $m1
	st32 $m0, $m6, $m15, 0
	sub $m4, $m4, 1
	brnz $m4, .Lcopy_u16
.Lcopy_u16_done:
	br $m10
	.size ipu_stack_static_copy_u16, .-ipu_stack_static_copy_u16

	.section .text.ipu_stack_static_patch_word,"ax",@progbits
	// Runtime helper ABI: m2 destination word, m3 value table, m4 remaining
	// iterations, m5 total iterations, m9 return. Exchange plans are ordinary
	// data consumed after this store; no instruction-cache maintenance is needed.
	.globl ipu_stack_static_patch_word
	.p2align 2
	.type ipu_stack_static_patch_word,@function
ipu_stack_static_patch_word:
	.supervisor
	sub $m5, $m5, $m4
	shl $m5, $m5, 2
	add $m3, $m3, $m5
	ld32 $m0, $m3, $m15, 0
	st32 $m0, $m2, $m15, 0
	br $m9
	.size ipu_stack_static_patch_word, .-ipu_stack_static_patch_word

	.section .text.ipu_stack_static_patch_row,"ax",@progbits
	// Runtime helper ABI: m2 row base, m3 offsets, m4 values, m5 word count,
	// m9 return. Patching occurs before the exchange synchronization boundary.
	.globl ipu_stack_static_patch_row
	.p2align 2
	.type ipu_stack_static_patch_row,@function
ipu_stack_static_patch_row:
	.supervisor
	brz $m5, .Lpatch_row_done
.Lpatch_row_next:
	ld32 $m6, $m3, $m15, 0
	ld32 $m0, $m4, $m15, 0
	add $m6, $m6, $m2
	st32 $m0, $m6, $m15, 0
	add $m3, $m3, 4
	add $m4, $m4, 4
	sub $m5, $m5, 1
	brnz $m5, .Lpatch_row_next
.Lpatch_row_done:
	br $m9
	.size ipu_stack_static_patch_row, .-ipu_stack_static_patch_row

	.section .text.ipu_stack_copy_u64,"ax",@progbits
	// Runtime helper ABI: m2 destination, m3 source, m4 words/worker, m5
	// remainder workers, m10 return. This is not a layout-conversion kernel.
	.globl ipu_stack_copy_u64
	.p2align 2
	.type ipu_stack_copy_u64,@function
ipu_stack_copy_u64:
	.supervisor
	add $m11, $m11, -16
	st32 $m2, $m11, $m15, 0
	st32 $m3, $m11, $m15, 1
	st32 $m4, $m11, $m15, 2
	st32 $m5, $m11, $m15, 3
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
	brz $m5, .Lcopy_u64_worker_done
.Lcopy_u64_worker_loop:
	ld64 $a0:1, $m3, $m15, 0
	st64 $a0:1, $m2, $m15, 0
	add $m3, $m3, 8
	add $m2, $m2, 8
	sub $m5, $m5, 1
	brnz $m5, .Lcopy_u64_worker_loop
.Lcopy_u64_worker_done:
	exitz $m15

	.section .text.ipu_stack_static_repeat_call,"ax",@progbits
	// Compact call loop used by static host schedules. ABI: m2 iteration count,
	// m3 body address, and m9 caller return address; the body returns in m10 and
	// must preserve m2. General tile-program repeats keep their counter on the
	// supervisor stack because ordinary kernel calls use m2 for their output.
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
	setzi $m7, ipu_stack_static_host_staging
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
	setzi $m5, ipu_stack_static_worker_stack_base
	st32 $m2, $m5, $m15, 0
	setzi $m0, ipu_stack_static_sample_cycle_worker
	runall $m0, $m5, 0
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	br $m10
	.size ipu_stack_static_sample_cycle, .-ipu_stack_static_sample_cycle

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

	.section .text.ipu_stack_static_assert_equal_u32,"ax",@progbits
	// Supervisor diagnostic helper: m3 actual, m4 expected, m5 word count.
	// A mismatch intentionally remains observable as a device timeout.
	.globl ipu_stack_static_assert_equal_u32
	.p2align 2
	.type ipu_stack_static_assert_equal_u32,@function
ipu_stack_static_assert_equal_u32:
	.supervisor
.Lassert_equal_u32_next:
	ld32 $m0, $m3, $m15, 0
	ld32 $m1, $m4, $m15, 0
	xor $m0, $m0, $m1
	brnz $m0, .Lassert_equal_u32_failed
	add $m3, $m3, 4
	add $m4, $m4, 4
	sub $m5, $m5, 1
	brnz $m5, .Lassert_equal_u32_next
	br $m10
.Lassert_equal_u32_failed:
	bri .Lassert_equal_u32_failed
	.size ipu_stack_static_assert_equal_u32, .-ipu_stack_static_assert_equal_u32

	.section .text.ipu_stack_static_worker_delay,"ax",@progbits
	// Diagnostic compute phase: m4 is the per-worker loop count. The generic
	// compute ABI's output and input pointers in m2/m3 are intentionally unused.
	.globl ipu_stack_static_worker_delay
	.p2align 2
	.type ipu_stack_static_worker_delay,@function
ipu_stack_static_worker_delay:
	.supervisor
	add $m11, $m11, -8
	st32 $m4, $m11, $m15, 0
	setzi $m0, .Lworker_delay
	runall $m0, $m11, 0
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	add $m11, $m11, 8
	br $m10
	.size ipu_stack_static_worker_delay, .-ipu_stack_static_worker_delay

	.worker
	.p2align 2
.Lworker_delay:
	ld32 $m0, $mvertex_base, $m15, 0
.Lworker_delay_next:
	sub $m0, $m0, 1
	brnz $m0, .Lworker_delay_next
	exitz $m15

	.section .text.ipu_stack_static_complete,"ax",@progbits
	.globl ipu_stack_static_complete
	.p2align 2
	.type ipu_stack_static_complete,@function
ipu_stack_static_complete:
	.supervisor
	setzi $m0, 1
	setzi $m1, ipu_stack_static_completion
	st32 $m0, $m1, $m15, 0
	zero $m0
	br $m0
	.size ipu_stack_static_complete, .-ipu_stack_static_complete
