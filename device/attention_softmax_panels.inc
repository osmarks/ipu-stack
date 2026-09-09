// Shared panel arithmetic for whole-row and segmented softmax workers.
	.macro SOFTMAX_MAX_PANEL
	mov $a7, $a6
	ld64step $a0:1, $mzero, $m3+=, 1
	.rept 3
	{ ld64step $a0:1, $mzero, $m3+=, 1
	  f16v4max $a6:7, $a6:7, $a0:1 }
	.endr
	f16v4max $a6:7, $a6:7, $a0:1
	f16v2max $a6, $a6, $a7
	.endm
	.macro SOFTMAX_STORE_QUAD
	f16v2exp $a2, $a2
	f16v2exp $a3, $a3
	{ st32step $a2, $mzero, $m2+=, 1
	  f16v2add $a6, $a6, $a2 }
	{ st32step $a3, $mzero, $m2+=, 1
	  f16v2add $a6, $a6, $a3 }
	.endm
	.macro SOFTMAX_EXP_PANEL
	// Sum only eight nonnegative exponentials per FP16 lane (each <= 1),
	// then accumulate the panel in FP32. Store the unchanged exponentials.
	// Prime with zero accumulators so the discarded readout is finite.
	// MIX then returns each preceding quad while starting the next one.
	setzi $a0, 1 << CSR_W_FP_CLR__ZAACC__SHIFT
	uput $FP_CLR, $a0
	ld64step $a0:1, $mzero, $m3+=, 1
	f16v4mix $a2:3, $a0:1, $a4:5
	.rept 3
	ld64step $a0:1, $mzero, $m3+=, 1
	f16v4mix $a2:3, $a0:1, $a4:5
	SOFTMAX_STORE_QUAD
	.endr
	f16v4gacc $a2:3
	SOFTMAX_STORE_QUAD
	f16v2tof32 $a0:1, $a6
	f32add $a0, $a0, $a1
	f32add $a7, $a7, $a0
	setzi $a6, 0
	.endm
	.macro SOFTMAX_EXP_PAIR
	f16v2tof32 $a0:1, $a0
	f32v2mul $a0:1, $a4:B, $a0:1
	f32v2add $a0:1, $a5:B, $a0:1
	f32v2tof16 $a0, $a0:1
	f16v2exp $a0, $a0
	{ st32step $a0, $mzero, $m2+=, 1
	  f16v2tof32 $a2:3, $a0 }
	.endm
