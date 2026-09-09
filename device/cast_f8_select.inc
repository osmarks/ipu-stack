// One descriptor and worker loop for every batch matrix. panelRows retains
// the flattened output-panel stride; row bounds limit this launch's rows.
	setzi $m0, WORKER_CODELET_SYMBOL
	brz $m9, 2f
	brz $m7, 2f
	brz $m5, 2f
	shr $m8, $m5, 16
	sub $m7, $m7, $m8
	brz $m7, 3f
	mul $m6, $m8, $m9
	shl $m6, $m6, 1
	shl $m5, $m8, 5
	bri 3f
2:
	setzi $m7, 0
3:
