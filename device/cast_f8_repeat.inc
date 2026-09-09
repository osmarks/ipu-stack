	brz $m7, 2f
	add $m3, $m3, $m6
	add $m2, $m2, $m5
	st32 $m3, $m11, $m15, 0
	st32 $m2, $m11, $m15, 1
	sub $m7, $m7, $m8
	bri 1b
2:
