	.text

#ifndef IPU_STACK_WORKER_BYTES
#define IPU_STACK_WORKER_BYTES 0x100
#endif

	.globl __worker_stack_max_growth_plus_one
	.set __worker_stack_max_growth_plus_one, IPU_STACK_WORKER_BYTES + 1

	.section .text._stackoverflow,"ax",@progbits
	.globl _stackoverflow
	.p2align 2
	.type _stackoverflow,@function
_stackoverflow:
	.worker
	trap 0
	.size _stackoverflow, .-_stackoverflow
