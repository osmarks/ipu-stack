	.text
	.allow_optimizations

	.set TILE_MUX_HOST, 0x600
	.set TILE_MUX_EXCHANGE, 0x640
	.set WORKER_BASE_PLACEHOLDER, 0
	.set WORKER_BASE_STRIDE, 0x30
	.set HOST_RUN_DESCRIPTOR_BYTES, 7 * 4
	.set SYNC_ALL, 7
	.set SYNC_COMPUTE_SET, 2
	.set SYNC_SUPERVISOR, 3
	.set OPCODE_SYNC, 0x41800000
	.set PROGRAM_ADDRESS_PLACEHOLDER, 0
	.set COMPLETION_ADDRESS_PLACEHOLDER, 0

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
	exitnz $m15
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

	.section .text.ipu_stack_static_host_run,"ax",@progbits
	.globl ipu_stack_static_host_run
	.p2align 2
	.type ipu_stack_static_host_run,@function
ipu_stack_static_host_run:
	.supervisor
.Lhost_run_next:
	ld32 $m5, $m3, $m15, 6
	brz $m5, .Lhost_run_call
	ld32 $m6, $m3, $m15, 4
	ld32 $m7, $m3, $m15, 5
.Lhost_run_packet_copy:
	ld32 $m0, $m7, $m15, 0
	st32 $m0, $m6, $m15, 0
	add $m7, $m7, 4
	add $m6, $m6, 4
	sub $m5, $m5, 1
	brnz $m5, .Lhost_run_packet_copy
.Lhost_run_call:
	st32 $m2, $m4, $m15, 0
	st32 $m3, $m4, $m15, 1
	ld32 $m0, $m3, $m15, 0
	setzi $m10, .Lhost_run_return
	br $m0
.Lhost_run_return:
	ld32 $m2, $m4, $m15, 0
	ld32 $m3, $m4, $m15, 1
	ld32 $m5, $m3, $m15, 3
	brz $m5, .Lhost_run_advance
	ld32 $m6, $m3, $m15, 1
	ld32 $m7, $m3, $m15, 2
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
