// Three contiguous panel segments per row; scratch follows maxima/sums.

	.macro SOFTMAX_SPLIT_INIT
	ld32 $m5, $mvertex_base, $m15, 2
	get $m9, $WSR
	and $m9, $m9, CSR_W_WSR__CTXTID_M1__MASK
	setzi $m0, 3
	cmpult $m4, $m9, $m0
	brnz $m4, .Lsoftmax_even_\@
	sub $m9, $m9, 3
	setzi $m4, 1
	bri .Lsoftmax_parity_\@
.Lsoftmax_even_\@:
	setzi $m4, 0
.Lsoftmax_parity_\@:
	add $m8, $m5, -1
	shl $m8, $m8, 5
	.endm

	.macro SOFTMAX_SPLIT_ROW
	ld32 $m2, $mvertex_base, $m15, 0
	ld32 $m3, $mvertex_base, $m15, 1
	// Three FP32 maxima and three sums use the existing extra 16 halves/row.
	setzi $m0, ATTENTION_KEY_BLOCK_COLUMNS * 2 + 8
	mul $m6, $m5, $m0
	add $m6, $m6, $m2
	setzi $m0, 3
	mul $m0, $m4, $m0
	add $m0, $m0, $m9
	shl $m0, $m0, 2
	add $m6, $m6, $m0
	setzi $m0, ((ATTENTION_KEY_BLOCK_COLUMNS + 47) / 48) * 16
	mul $m0, $m9, $m0
	ld32 $m10, $mvertex_base, $m15, 3
	sub $m10, $m10, $m0
	cmpslt $m1, $m10, $mzero
	brz $m1, .Lsoftmax_nonempty_\@
	setzi $m10, 0
.Lsoftmax_nonempty_\@:
	// Capacity of this segment; m11 remains the worker stack pointer.
	setzi $m1, ATTENTION_KEY_BLOCK_COLUMNS
	sub $m1, $m1, $m0
	setzi $m7, ((ATTENTION_KEY_BLOCK_COLUMNS + 47) / 48) * 16
	cmpult $m7, $m7, $m1
	brz $m7, .Lsoftmax_capacity_\@
	setzi $m1, ((ATTENTION_KEY_BLOCK_COLUMNS + 47) / 48) * 16
.Lsoftmax_capacity_\@:
	cmpult $m7, $m1, $m10
	brz $m7, .Lsoftmax_clamped_\@
	mov $m10, $m1
.Lsoftmax_clamped_\@:
	mul $m0, $m0, $m5
	shl $m0, $m0, 1
	shl $m1, $m4, 5
	add $m0, $m0, $m1
	add $m2, $m2, $m0
	add $m3, $m3, $m0
	.endm

	.worker
	.p2align 3
.Lsoftmax_split_max:
	SOFTMAX_SPLIT_INIT
.Lsoftmax_split_max_row:
	cmpult $m0, $m4, $m5
	brz $m0, .Lsoftmax_done
	SOFTMAX_SPLIT_ROW
	ldconst $a6, 0xfbfffbff
	shr $m7, $m10, 4
	brz $m7, .Lsoftmax_split_max_tail
	sub $m7, $m7, 1
.Lsoftmax_split_max_panel:
	SOFTMAX_MAX_PANEL
	add $m3, $m3, $m8
	brnzdec $m7, .Lsoftmax_split_max_panel
.Lsoftmax_split_max_tail:
	and $m7, $m10, 15
	brz $m7, .Lsoftmax_split_max_reduce
	and $m1, $m7, 1
	shr $m7, $m7, 1
	brz $m7, .Lsoftmax_split_max_odd
	sub $m7, $m7, 1
.Lsoftmax_split_max_pair:
	ld32step $a0, $mzero, $m3+=, 1
	f16v2max $a6, $a6, $a0
	brnzdec $m7, .Lsoftmax_split_max_pair
.Lsoftmax_split_max_odd:
	brz $m1, .Lsoftmax_split_max_reduce
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32tof16 $a0, $a0
	f16v2max $a6, $a6, $a0
.Lsoftmax_split_max_reduce:
	f16v2tof32 $a6:7, $a6
	f32max $a6, $a6, $a7
	st32 $a6, $m6, $m15, 0
	add $m4, $m4, 2
	bri .Lsoftmax_split_max_row

	.p2align 3
.Lsoftmax_split_exp:
	SOFTMAX_SPLIT_INIT
.Lsoftmax_split_exp_row:
	cmpult $m0, $m4, $m5
	brz $m0, .Lsoftmax_done
	SOFTMAX_SPLIT_ROW
	shl $m0, $m9, 2
	sub $m0, $m6, $m0
	ld32 $a0, $m0, $m15, 0
	ld32 $a1, $m0, $m15, 1
	ld32 $a2, $m0, $m15, 2
	f32max $a0, $a0, $a1
	f32max $a0, $a0, $a2
	f32tof16 $a4, $a0
	mov $a5, $a4
	brnz $m9, .Lsoftmax_split_scale
	ldconst $a1, ATTENTION_SCALE_BITS
	f32mul $a2, $a0, $a1
	ld32 $m0, $mvertex_base, $m15, 0
	setzi $m1, ATTENTION_KEY_BLOCK_COLUMNS * 2
	mul $m1, $m5, $m1
	add $m0, $m0, $m1
	shl $m1, $m4, 2
	add $m0, $m0, $m1
	st32 $a2, $m0, $m15, 0
.Lsoftmax_split_scale:
	ldconst $a0, ATTENTION_SCALE_BITS
	f32sub $a1, $azero, $a0
	f32v2tof16 $a0, $a0:1
	uput $TAS, $a0
	setzi $a6, 0
	setzi $a7, 0
	shr $m7, $m10, 4
	brz $m7, .Lsoftmax_split_exp_tail
	sub $m7, $m7, 1
.Lsoftmax_split_exp_panel:
	SOFTMAX_EXP_PANEL
	add $m3, $m3, $m8
	add $m2, $m2, $m8
	brnzdec $m7, .Lsoftmax_split_exp_panel
.Lsoftmax_split_exp_tail:
	and $m7, $m10, 15
	brz $m7, .Lsoftmax_split_padding
	f16v2tof32 $a2:3, $a4
	ldconst $a4, ATTENTION_SCALE_BITS
	f32mul $a5, $a2, $a4
	f32sub $a5, $azero, $a5
	shr $m7, $m7, 1
	brz $m7, .Lsoftmax_split_exp_odd
	sub $m7, $m7, 1
.Lsoftmax_split_exp_pair:
	ld32step $a0, $mzero, $m3+=, 1
	SOFTMAX_EXP_PAIR
	f32v2add $a6:7, $a6:7, $a2:3
	brnzdec $m7, .Lsoftmax_split_exp_pair
.Lsoftmax_split_exp_odd:
	and $m0, $m10, 1
	brz $m0, .Lsoftmax_split_zero_tail
	ld32 $a0, $m3, $m15, 0
	f16v2tof32 $a0:1, $a0
	f32v2mul $a0:1, $a4:B, $a0:1
	f32v2add $a0:1, $a5:B, $a0:1
	ldconst $a1, 0xc77fe000
	f32v2tof16 $a0, $a0:1
	f16v2exp $a0, $a0
	{ st32step $a0, $mzero, $m2+=, 1
	  f16v2tof32 $a2:3, $a0 }
	f32v2add $a6:7, $a6:7, $a2:3
.Lsoftmax_split_zero_tail:
	and $m7, $m10, 15
	add $m7, $m7, 1
	shr $m7, $m7, 1
	setzi $m0, 8
	sub $m7, $m0, $m7
	brz $m7, .Lsoftmax_split_after_tail
	sub $m7, $m7, 1
.Lsoftmax_split_zero_pair:
	st32step $mzero, $mzero, $m2+=, 1
	brnzdec $m7, .Lsoftmax_split_zero_pair
.Lsoftmax_split_after_tail:
	add $m2, $m2, $m8
.Lsoftmax_split_padding:
	setzi $m7, ((ATTENTION_KEY_BLOCK_COLUMNS + 47) / 48) * 16
	mul $m0, $m9, $m7
	setzi $m1, ATTENTION_KEY_BLOCK_COLUMNS
	sub $m1, $m1, $m0
	cmpult $m0, $m7, $m1
	brz $m0, .Lsoftmax_split_padding_capacity
	mov $m1, $m7
.Lsoftmax_split_padding_capacity:
	shr $m7, $m1, 4
	add $m0, $m10, 15
	shr $m0, $m0, 4
	sub $m7, $m7, $m0
	brz $m7, .Lsoftmax_split_store_sum
	sub $m7, $m7, 1
.Lsoftmax_split_zero_panel:
	.rept 8
	st32step $mzero, $mzero, $m2+=, 1
	.endr
	add $m2, $m2, $m8
	brnzdec $m7, .Lsoftmax_split_zero_panel
.Lsoftmax_split_store_sum:
	setzi $m0, 12
	mul $m0, $m5, $m0
	add $m6, $m6, $m0
	f32add $a6, $a6, $a7
	st32 $a6, $m6, $m15, 0
	add $m4, $m4, 2
	bri .Lsoftmax_split_exp_row

	.p2align 3
.Lsoftmax_split_sum:
	ld32 $m5, $mvertex_base, $m15, 2
	ld32 $m2, $mvertex_base, $m15, 0
	setzi $m0, ATTENTION_KEY_BLOCK_COLUMNS * 2 + 4
	mul $m0, $m5, $m0
	add $m2, $m2, $m0
	get $m4, $WSR
	and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
.Lsoftmax_split_sum_row:
	cmpult $m0, $m4, $m5
	brz $m0, .Lsoftmax_done
	shl $m0, $m4, 2
	add $m3, $m2, $m0
	setzi $m0, 16
	mul $m0, $m5, $m0
	add $m6, $m2, $m0
	setzi $m0, 12
	mul $m0, $m4, $m0
	add $m6, $m6, $m0
	ld32 $a0, $m6, $m15, 0
	ld32 $a1, $m6, $m15, 1
	ld32 $a2, $m6, $m15, 2
	f32add $a0, $a0, $a1
	f32add $a0, $a0, $a2
	st32 $a0, $m3, $m15, 0
	add $m4, $m4, 6
	bri .Lsoftmax_split_sum_row
