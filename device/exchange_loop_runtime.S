	.text
	.allow_optimizations

	.section .text.ipu_stack_exchange_loop_start,"ax",@progbits
	.globl ipu_stack_exchange_loop_start
	.p2align 2
	nop
	.type ipu_stack_exchange_loop_start,@function
ipu_stack_exchange_loop_start:
	.supervisor
	.long 0x41800003
	setzi $m0, 0x640
	put 0xa0, $m0
	put 0xa7, $m15
	setzi $m11, 0x50000
	put 0x21, $m11
	setzi $m11, 0x50030
	put 0x22, $m11
	setzi $m11, 0x50060
	put 0x23, $m11
	setzi $m11, 0x50090
	put 0x24, $m11
	setzi $m11, 0x500c0
	put 0x25, $m11
	setzi $m11, 0x500f0
	put 0x26, $m11
	setzi $m11, 0x50120
	setzi $m0, ipu_stack_exchange_loop_worker_init
	runall $m0, $m15, 0
	setzi $m0, 0x600
	put 0xa0, $m0
	.long 0x41800007
	setzi $m0, 0x640
	put 0xa0, $m0
	.globl ipu_stack_command_table_address
ipu_stack_command_table_address:
	setzi $m6, 0x59000

.Lcommand_loop:
	ld32 $m5, $m6, $m15, 0
	brz $m5, .Ldone
	call $m7, ipu_stack_pre_sync_dispatch
	ld32 $m1, $m6, $m15, 1
	ld32 $m2, $m6, $m15, 2
	ld32 $m3, $m6, $m15, 3
	add $m6, $m6, 16
	cmpeq $m4, $m5, 1
	brnz $m4, .Llaunch_complete
	cmpeq $m4, $m5, 2
	brnz $m4, .Linactive_launch
	cmpeq $m4, $m5, 3
	brz $m4, .Lset_receive_base
	put 0xa7, $m2
	bri .Larm_plan
.Lset_receive_base:
	put 0xa4, $m3
.Larm_plan:
	setzi $m8, 1
	put 0xa6, $m8
	setzi $m10, .Lplan_complete
	br $m1

.Lplan_complete:
	cmpeq $m4, $m5, 4
	brz $m4, .Llaunch_complete
	ld32 $m8, $m2, $m15, 0
	ld32 $m9, $m3, $m15, 0
	add $m8, $m8, $m9
	st32 $m8, $m2, $m15, 0
#ifdef IPU_STACK_TRAP_AFTER_RECEIVE
	zero $m0
	br $m0
#endif
	bri .Llaunch_complete

.Linactive_launch:
	.long 0x40c00000
	.long 0x41800001

.Llaunch_complete:
	setzi $m0, 0x640
	put 0xa0, $m0
	bri .Lcommand_loop

.Ldone:
	zero $m0
	br $m0
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
	.long 0x40c00001
	.long 0x41800001
	br $m7

	.globl ipu_stack_global_pre_sync
ipu_stack_global_pre_sync:
	call $m10, .Lglobal_sync_wait
	call $m10, .Lglobal_sync_send
	call $m10, .Lglobal_sync_local
	call $m10, .Lglobal_sync_release
	br $m7
.Lglobal_sync_local:
	.long 0x41800007
	setzi $m8, 0x640
	put 0xa0, $m8
	br $m10
.Lglobal_sync_send:
	setzi $m8, 1
	put 0xa6, $m8
	.globl ipu_stack_loop_global_sync_send0
ipu_stack_loop_global_sync_send0:
	.long 0x78200003
	.globl ipu_stack_loop_global_sync_send1
ipu_stack_loop_global_sync_send1:
	.long 0x78200003
	.long 0x41800000
	br $m10
.Lglobal_sync_wait:
	.globl ipu_stack_loop_global_sync_endpoint
ipu_stack_loop_global_sync_endpoint:
	setzi $m8, 0x600
	put 0xa0, $m8
	.long 0x4180000f
	br $m10
.Lglobal_sync_release:
	.long 0x41800003
	.long 0x40a00048
	.globl ipu_stack_loop_global_sync_release
ipu_stack_loop_global_sync_release:
	.long 0x78000003
	br $m10
