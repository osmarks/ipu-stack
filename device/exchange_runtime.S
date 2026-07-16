	.text
	.allow_optimizations

	.section .text.ipu_stack_exchange_start,"ax",@progbits
	.globl ipu_stack_exchange_start
	.p2align 2
	nop
	.type ipu_stack_exchange_start,@function
ipu_stack_exchange_start:
	.supervisor
	# Secondary-loader application handoff rendezvous.
	.long 0x41800003
	setzi $m0, 0x640
	put 0xa0, $m0
	put 0xa7, $m15
	# Initialize the six workers' internal-sync endpoint registers.
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
	setzi $m0, ipu_stack_exchange_worker_init
	runall $m0, $m15, 0
	# Pair-safe local rendezvous used by the SDK before any global sync. Without
	# this, paired tiles can update INCOMING_MUXPAIR concurrently and raise
	# TEXCPT_EXCONF.
	setzi $m0, 0x600
	put 0xa0, $m0
.Linitial_sync:
	.long 0x41800007
	setzi $m0, 0x640
	put 0xa0, $m0
	# program::Sync(INTERNAL), recovered from the SDK supervisor dispatcher.
	.globl ipu_stack_pre_exchange_sync
ipu_stack_pre_exchange_sync:
	call $m7, ipu_stack_global_pre_sync
	call $m10, ipu_stack_execute
.Lexchange_complete:
	.globl ipu_stack_post_exchange_sync
ipu_stack_post_exchange_sync:
#ifdef IPU_STACK_TRAP_BEFORE_DCOUNT_CLEAR
	zero $m0
	br $m0
#endif
	zero $m0
	put 0xa6, $m0
	setzi $m0, 0x640
	put 0xa0, $m0
#ifdef IPU_STACK_TRAP_AFTER_EXCHANGE
	zero $m0
	br $m0
#endif
.Lidle:
	bri .Lidle
	.globl ipu_stack_nonmaster_redirect
ipu_stack_nonmaster_redirect:
	bri ipu_stack_nonmaster_sync
	.globl ipu_stack_nonmaster_sync
ipu_stack_nonmaster_sync:
	.long 0x40c00001
	.long 0x41800001
	call $m10, ipu_stack_execute
	bri ipu_stack_post_exchange_sync
	.size ipu_stack_exchange_start, .-ipu_stack_exchange_start

	.section .text.ipu_stack_exchange_worker_init,"ax",@progbits
	.globl ipu_stack_exchange_worker_init
	.p2align 2
	.type ipu_stack_exchange_worker_init,@function
ipu_stack_exchange_worker_init:
	.worker
	.globl ipu_stack_worker_sync_base
ipu_stack_worker_sync_base:
	# Patched per physical tile to logical_tile * 8 by the Rust packager.
	setzi $m0, 0
	get $m1, 1
	and $m1, $m1, 7
	or $m0, $m0, $m1
	st32 $m0, $m12, $m15, 0
	ld32 $a0, $m12, $m15, 0
	put 0x7, $a0
	exitnz $m15
	.size ipu_stack_exchange_worker_init, .-ipu_stack_exchange_worker_init

	.section .text.ipu_stack_global_pre_sync,"ax",@progbits
	.p2align 2
	.globl ipu_stack_execute
	.type ipu_stack_execute,@function
ipu_stack_execute:
#ifdef IPU_STACK_TRAP_BEFORE_EXCHANGE
	zero $m0
	br $m0
#endif
	setzi $m8, 1
	put 0xa6, $m8
	setzi $m0, 0x54000
	setzi $m10, .Lexchange_complete
	br $m0
	.size ipu_stack_execute, .-ipu_stack_execute
	.globl ipu_stack_nonparticipant_redirect
ipu_stack_nonparticipant_redirect:
	bri ipu_stack_nonparticipant_execute
	.globl ipu_stack_coordinator_nonparticipant_redirect
ipu_stack_coordinator_nonparticipant_redirect:
	br $m10
	.globl ipu_stack_nonparticipant_execute
ipu_stack_nonparticipant_execute:
	# Exchange-launch participation for tiles with no sender/receiver plan.
	.long 0x40c00000
	.long 0x41800001
	br $m10
	.p2align 3
	.globl ipu_stack_global_pre_sync
	.type ipu_stack_global_pre_sync,@function
ipu_stack_global_pre_sync:
	# Other tiles replace this routine with the SDK's `sans 1; sync 1`
	# receive path at package time.
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
	.globl ipu_stack_global_sync_send0
ipu_stack_global_sync_send0:
	.long 0x78200003
	.globl ipu_stack_global_sync_send1
ipu_stack_global_sync_send1:
	.long 0x78200003
	.long 0x41800000
	br $m10
.Lglobal_sync_wait:
	.globl ipu_stack_global_sync_endpoint
ipu_stack_global_sync_endpoint:
	setzi $m8, 0x600
	put 0xa0, $m8
#ifdef IPU_STACK_TRAP_AFTER_WORKER_INIT
	get $m0, 0xff
#endif
	.long 0x4180000f
	br $m10
.Lglobal_sync_release:
	.long 0x41800003
	.long 0x40a00048
	.globl ipu_stack_global_sync_release
ipu_stack_global_sync_release:
	.long 0x78000003
	br $m10
	.size ipu_stack_global_pre_sync, .-ipu_stack_global_pre_sync
