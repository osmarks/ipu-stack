// Shared cast ABI: m5 row bounds, m7 packed rows, m9 row-major stride.
// Select once per tile rather than branching/spilling in every worker.
	setzi $m0, WORKER_CODELET_SYMBOL
	brz $m9, 1f
	brz $m7, 1f
	brz $m5, 1f
	shr $m1, $m5, 16
	cmpeq $m1, $m1, $m7
	brnz $m1, 1f
	shl $m1, $m5, 16
	shr $m1, $m1, 16
	shr $m0, $m5, 16
	cmpeq $m1, $m0, $m1
	setzi $m0, WORKER_CODELET_SYMBOL
	brnz $m1, 1f
	setzi $m0, __runCodelet_Cast2To1Padded
1:
