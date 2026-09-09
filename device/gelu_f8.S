#define GELU_MACROS_ONLY
#include "gelu_f16.S"

// output, input, rows, width (multiple of four), scale, packed output, input row stride, output columns.
.text
.allow_optimizations
.section .text.gelu_f8,"ax",@progbits
.globl gelu_f8
.p2align 2
.type gelu_f8,@function
gelu_f8:
.supervisor
add $m11, $m11, -40
st32 $m2, $m11, $m15, 0
st32 $m3, $m11, $m15, 1
st32 $m4, $m11, $m15, 2
ldconst $m0, 0x48004800
st32 $m0, $m11, $m15, 3
ldconst $m0, 0xc800c800
st32 $m0, $m11, $m15, 4
sub $m6, 0, $m6
and $m6, $m6, 255
st32 $m6, $m11, $m15, 5
st32 $m5, $m11, $m15, 6
st32 $m7, $m11, $m15, 7
st32 $m8, $m11, $m15, 8
st32 $m9, $m11, $m15, 9
setzi $m0, .Lfp8_gelu_worker
runall $m0, $m11, 0
sync SYNC_COMPUTE_SET
add $m11, $m11, 40
br $m10
.size gelu_f8, .-gelu_f8

.macro GELU_FP8_QUAD offset, output_offset
ld32 $a0, $m3, $m15, \offset
ld32 $a1, $m3, $m15, (\offset + 1)
GELU_TWO_PAIRS \offset
f16v2tof8 $a2, $a2
f16v2tof8 $a3, $a3
sort4x16lo $a2, $a2, $a3
st32 $a2, $m2, $m15, \output_offset
.endm

.worker
.p2align 3
.Lfp8_gelu_worker:
setzi $a0, 2
put 9, $a0
ld32 $a0, $mvertex_base, $m15, 5
put 10, $a0
LOAD_GELU_CONSTANTS
setzi $m1, 0
ld32 $m7, $mvertex_base, $m15, 2
.Lfp8_gelu_row:
ld32 $m8, $mvertex_base, $m15, 6
ld32 $m3, $mvertex_base, $m15, 1
ld32 $m0, $mvertex_base, $m15, 8
mul $m0, $m1, $m0
shl $m0, $m0, 1
add $m3, $m3, $m0
get $m4, $WSR
and $m4, $m4, CSR_W_WSR__CTXTID_M1__MASK
shl $m4, $m4, 5
shl $m0, $m4, 1
add $m3, $m3, $m0
ld32 $m2, $mvertex_base, $m15, 0
ld32 $m0, $mvertex_base, $m15, 7
brz $m0, .Lfp8_gelu_linear
mul $m0, $m4, $m7
add $m2, $m2, $m0
shl $m0, $m1, 5
add $m2, $m2, $m0
setzi $m0, 192
mul $m9, $m7, $m0
bri .Lfp8_gelu_loop
.Lfp8_gelu_linear:
ld32 $m0, $mvertex_base, $m15, 9
mul $m0, $m1, $m0
add $m2, $m2, $m0
add $m2, $m2, $m4
setzi $m9, 192
.Lfp8_gelu_loop:
add $m0, $m4, 32
cmpult $m6, $m8, $m0
brnz $m6, .Lfp8_gelu_tail
GELU_FP8_QUAD 0, 0
GELU_FP8_QUAD 2, 1
GELU_FP8_QUAD 4, 2
GELU_FP8_QUAD 6, 3
GELU_FP8_QUAD 8, 4
GELU_FP8_QUAD 10, 5
GELU_FP8_QUAD 12, 6
GELU_FP8_QUAD 14, 7
add $m3, $m3, 384
add $m2, $m2, $m9
add $m4, $m4, 192
bri .Lfp8_gelu_loop
.Lfp8_gelu_tail:
cmpult $m6, $m4, $m8
brz $m6, .Lfp8_gelu_padding
GELU_FP8_QUAD 0, 0
add $m3, $m3, 8
add $m2, $m2, 4
add $m4, $m4, 4
bri .Lfp8_gelu_tail
.Lfp8_gelu_padding:
ld32 $m8, $mvertex_base, $m15, 9
.Lfp8_gelu_zero:
cmpult $m6, $m4, $m8
brz $m6, .Lfp8_gelu_next_row
st32 $m15, $m2, $m15, 0
add $m2, $m2, 4
add $m4, $m4, 4
and $m0, $m4, 31
brnz $m0, .Lfp8_gelu_zero
// The next panel assigned to this worker is six panels ahead.
add $m4, $m4, 160
add $m2, $m2, $m9
add $m2, $m2, -32
bri .Lfp8_gelu_zero
.Lfp8_gelu_next_row:
add $m1, $m1, 1
cmpult $m6, $m1, $m7
brnz $m6, .Lfp8_gelu_row
exitz $m15
