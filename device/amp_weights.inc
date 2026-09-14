// Source groups are output-major, inner-minor. The non-monotonic CWEI
// destinations match Poplin's IPU21 1x1 kernel and its accumulator drain order.
// Wider loads consume two adjacent slots; assembly still emits straight-line code.
	.macro LOAD_AMP16_WEIGHTS bits=64
	.irp group, 0, 32, 8, 40, 16, 48, 24, 56
	.irp slot, 0, 1, 2, 3, 4, 5, 6, 7
	.if \slot % (\bits / 64) == 0
	ld\bits\()putcs (\group + \slot)
	.endif
	.endr
	.endr
	.endm
