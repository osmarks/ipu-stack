	.text
	.allow_optimizations

	.set ROLE_INACTIVE, 2
	.set ROLE_EXCHANGE_ABSOLUTE, 10
	.set ROLE_COMPUTE, 11
	.set ROLE_COMPUTE_NOOP, 12
	.set ROLE_HOST_PROGRAM, 13
	.set ROLE_HOST_IDLE, 14
	.set TILE_MUX_HOST, 0x600
	.set TILE_MUX_EXCHANGE, 0x640
	.set WORKER_BASE_START, 0x50000
	.set WORKER_BASE_STRIDE, 0x30
	.set SYNC_ALL, 7
	.set SYNC_COMPUTE_SET, 2
	.set SYNC_SUPERVISOR, 3
	.set SYNC_HOST, 15
	.set SYNC_EXTERNAL, 1
	.set SYNC_RECEIVE, 0
	.set SANS_INACTIVE, 0
	.set SANS_EXTERNAL, 1
	.set OPCODE_SYNC, 0x41800000
	.set OPCODE_SANS, 0x40c00000
	.set OPCODE_DELAY, 0x40a00000
	.set SEND_ONE_WORD_PLACEHOLDER, 0x78200003
	.set SEND_RELEASE_PLACEHOLDER, 0x78000003
	.set GLOBAL_SYNC_RELEASE_DELAY, 72
	.set COMMAND_TABLE_PLACEHOLDER, 0x59000
	.set COMPLETION_ADDRESS_PLACEHOLDER, 0

	.section .text.ipu_stack_exchange_loop_start,"ax",@progbits
	.globl ipu_stack_exchange_loop_start
	.p2align 2
	nop
	.type ipu_stack_exchange_loop_start,@function
ipu_stack_exchange_loop_start:
	.supervisor
	.long OPCODE_SYNC | SYNC_SUPERVISOR
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	put 0xa7, $m15
	setzi $m11, WORKER_BASE_START
	put 0x21, $m11
	setzi $m11, WORKER_BASE_START + WORKER_BASE_STRIDE
	put 0x22, $m11
	setzi $m11, WORKER_BASE_START + 2 * WORKER_BASE_STRIDE
	put 0x23, $m11
	setzi $m11, WORKER_BASE_START + 3 * WORKER_BASE_STRIDE
	put 0x24, $m11
	setzi $m11, WORKER_BASE_START + 4 * WORKER_BASE_STRIDE
	put 0x25, $m11
	setzi $m11, WORKER_BASE_START + 5 * WORKER_BASE_STRIDE
	put 0x26, $m11
	setzi $m11, WORKER_BASE_START + 6 * WORKER_BASE_STRIDE
	setzi $m0, ipu_stack_exchange_loop_worker_init
	runall $m0, $m15, 0
	setzi $m0, TILE_MUX_HOST
	put 0xa0, $m0
	.long OPCODE_SYNC | SYNC_ALL
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	call $m7, ipu_stack_pre_sync_dispatch
	.globl ipu_stack_command_table_address
ipu_stack_command_table_address:
	setzi $m6, COMMAND_TABLE_PLACEHOLDER

.Lcommand_loop:
	ld32 $m5, $m6, $m15, 0
	brz $m5, .Ldone
	cmpeq $m8, $m5, ROLE_HOST_PROGRAM
	brnz $m8, .Lload_host_command
	cmpeq $m8, $m5, ROLE_HOST_IDLE
	brnz $m8, .Lload_host_command
	cmpeq $m8, $m5, ROLE_INACTIVE
	brnz $m8, .Lpre_sync_exchange
	cmpeq $m8, $m5, ROLE_EXCHANGE_ABSOLUTE
	brz $m8, .Llaunch_workers
.Lpre_sync_exchange:
	call $m7, ipu_stack_pre_sync_dispatch
.Llaunch_workers:
	setzi $m0, ipu_stack_exchange_loop_worker_barrier
	runall $m0, $m15, 0
	.long OPCODE_SYNC | SYNC_COMPUTE_SET
	ld32 $m1, $m6, $m15, 1
	ld32 $m2, $m6, $m15, 2
	ld32 $m3, $m6, $m15, 3
	ld32 $m4, $m6, $m15, 4
	add $m6, $m6, 20
	bri .Ldispatch_command
.Lload_host_command:
	ld32 $m1, $m6, $m15, 1
	add $m6, $m6, 20
.Ldispatch_command:
	cmpeq $m8, $m5, ROLE_INACTIVE
	brnz $m8, .Linactive_launch
	cmpeq $m8, $m5, ROLE_EXCHANGE_ABSOLUTE
	brnz $m8, .Lset_absolute_exchange_bases
	cmpeq $m8, $m5, ROLE_COMPUTE
	brnz $m8, .Lrun_compute
	cmpeq $m8, $m5, ROLE_COMPUTE_NOOP
	brnz $m8, .Llaunch_complete
	cmpeq $m8, $m5, ROLE_HOST_PROGRAM
	brnz $m8, .Lrun_host_program
	cmpeq $m8, $m5, ROLE_HOST_IDLE
	brnz $m8, .Lhost_idle
	bri .Lhalt
.Lset_absolute_exchange_bases:
	put 0xa7, $m15
	put 0xa4, $m15
.Larm_plan:
	setzi $m8, 1
	put 0xa6, $m8
	setzi $m10, .Lplan_complete
	br $m1

.Lrun_compute:
	setzi $m10, .Llaunch_complete
	br $m1

.Lrun_host_program:
	setzi $m10, .Lhost_launch_complete
	br $m1

.Lhost_idle:
	.long OPCODE_SANS | SANS_EXTERNAL
	.long OPCODE_SYNC | SYNC_EXTERNAL
	bri .Lhost_launch_complete

.Lplan_complete:
#ifdef IPU_STACK_TRAP_AFTER_RECEIVE
	zero $m0
	br $m0
#endif
	bri .Llaunch_complete

.Linactive_launch:
	.long OPCODE_SANS | SANS_INACTIVE
	.long OPCODE_SYNC | SYNC_EXTERNAL
	bri .Llaunch_complete

.Llaunch_complete:
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	call $m7, ipu_stack_device_sync_dispatch
	bri .Lcommand_loop

.Lhost_launch_complete:
	setzi $m0, TILE_MUX_EXCHANGE
	put 0xa0, $m0
	call $m7, ipu_stack_device_sync_dispatch
	bri .Lcommand_loop

.Ldone:
	setzi $m0, 1
	.globl ipu_stack_completion_address
ipu_stack_completion_address:
	setzi $m1, COMPLETION_ADDRESS_PLACEHOLDER
	st32 $m0, $m1, $m15, 0
	.globl ipu_stack_completion_dispatch
ipu_stack_completion_dispatch:
	bri ipu_stack_master_completion
	.globl ipu_stack_nonmaster_completion_redirect
ipu_stack_nonmaster_completion_redirect:
	bri .Lhalt
	.globl ipu_stack_master_completion
ipu_stack_master_completion:
	bri .Lhalt
.Lhalt:
	bri .Lhalt
	.size ipu_stack_exchange_loop_start, .-ipu_stack_exchange_loop_start

	.section .text.ipu_stack_exchange_loop_worker_init,"ax",@progbits
	.globl ipu_stack_exchange_loop_worker_init
	.p2align 2
	.type ipu_stack_exchange_loop_worker_init,@function
ipu_stack_exchange_loop_worker_init:
	.worker
	.globl ipu_stack_loop_worker_sync_base
ipu_stack_loop_worker_sync_base:
	setzi $m0, 0
	get $m1, 1
	and $m1, $m1, 7
	or $m0, $m0, $m1
	st32 $m0, $m12, $m15, 0
	ld32 $a0, $m12, $m15, 0
	put 0x7, $a0
	exitnz $m15
	.size ipu_stack_exchange_loop_worker_init, .-ipu_stack_exchange_loop_worker_init

	.section .text.ipu_stack_exchange_loop_worker_barrier,"ax",@progbits
	.globl ipu_stack_exchange_loop_worker_barrier
	.p2align 2
	.type ipu_stack_exchange_loop_worker_barrier,@function
ipu_stack_exchange_loop_worker_barrier:
	.worker
	exitz $m15
	.size ipu_stack_exchange_loop_worker_barrier, .-ipu_stack_exchange_loop_worker_barrier

	.section .text.ipu_stack_exchange_loop_sync,"ax",@progbits
	.p2align 3
	.globl ipu_stack_pre_sync_dispatch
ipu_stack_pre_sync_dispatch:
	bri ipu_stack_global_pre_sync
	.globl ipu_stack_nonmaster_pre_sync_redirect
ipu_stack_nonmaster_pre_sync_redirect:
	bri ipu_stack_nonmaster_pre_sync
	.globl ipu_stack_nonmaster_pre_sync
ipu_stack_nonmaster_pre_sync:
	.long OPCODE_SANS | SANS_EXTERNAL
	.long OPCODE_SYNC | SYNC_EXTERNAL
	br $m7

	.globl ipu_stack_device_sync_dispatch
ipu_stack_device_sync_dispatch:
	bri ipu_stack_global_device_sync
	.globl ipu_stack_nonmaster_device_sync_redirect
ipu_stack_nonmaster_device_sync_redirect:
	bri ipu_stack_nonmaster_pre_sync
	.globl ipu_stack_global_device_sync
ipu_stack_global_device_sync:
	call $m10, .Lglobal_sync_select_endpoint
	call $m10, .Lglobal_sync_send
	call $m10, .Lglobal_sync_local
	call $m10, .Lglobal_sync_release
	br $m7

	.globl ipu_stack_global_pre_sync
ipu_stack_global_pre_sync:
	call $m10, .Lglobal_sync_wait
	call $m10, .Lglobal_sync_send
	call $m10, .Lglobal_sync_local
	call $m10, .Lglobal_sync_release
	br $m7
.Lglobal_sync_local:
	.long OPCODE_SYNC | SYNC_ALL
	setzi $m8, TILE_MUX_EXCHANGE
	put 0xa0, $m8
	br $m10
.Lglobal_sync_send:
	setzi $m8, 1
	put 0xa6, $m8
	.globl ipu_stack_loop_global_sync_send0
ipu_stack_loop_global_sync_send0:
	.long SEND_ONE_WORD_PLACEHOLDER
	.globl ipu_stack_loop_global_sync_send1
ipu_stack_loop_global_sync_send1:
	.long SEND_ONE_WORD_PLACEHOLDER
	.long OPCODE_SYNC | SYNC_RECEIVE
	br $m10
.Lglobal_sync_wait:
	call $m10, .Lglobal_sync_select_endpoint
	.long OPCODE_SYNC | SYNC_HOST
	br $m10
.Lglobal_sync_select_endpoint:
	.globl ipu_stack_loop_global_sync_endpoint
ipu_stack_loop_global_sync_endpoint:
	setzi $m8, TILE_MUX_HOST
	put 0xa0, $m8
	br $m10
.Lglobal_sync_release:
	.long OPCODE_SYNC | SYNC_SUPERVISOR
	.long OPCODE_DELAY | GLOBAL_SYNC_RELEASE_DELAY
	.globl ipu_stack_loop_global_sync_release
ipu_stack_loop_global_sync_release:
	.long SEND_RELEASE_PLACEHOLDER
	br $m10
