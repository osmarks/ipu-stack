// FP8 probability stores; maxima/denominators remain FP32. The packed
// matrix has 32-key panels, while score reads still use 16-key FP16 panels.
#ifdef ATTENTION_OUTPUT_F8
#define SOFTMAX_ELEMENT_BYTES 1
#else
#define SOFTMAX_ELEMENT_BYTES 2
#endif

.macro SOFTMAX_CONFIG_OUTPUT
#ifdef ATTENTION_OUTPUT_F8
setzi $a0, 2
put 9, $a0
setzi $a0, ((-ATTENTION_OUTPUT_SCALE) & 255)
put 10, $a0
#endif
.endm

.macro SOFTMAX_NEXT_OUTPUT
#ifdef ATTENTION_OUTPUT_F8
ld32 $m0, $mvertex_base, $m15, 0
sub $m0, $m2, $m0
and $m0, $m0, 31
brnz $m0, .Lsoftmax_half_panel_\@
add $m2, $m2, $m8
.Lsoftmax_half_panel_\@:
#else
add $m2, $m2, $m8
#endif
.endm

.macro SOFTMAX_FP8_STORE_EIGHT
f16v2add $a6, $a6, $a0
f16v2add $a6, $a6, $a1
f16v2add $a6, $a6, $a2
f16v2add $a6, $a6, $a3
f16v8tof8 $a0:1, $a0:3
st64step $a0:1, $mzero, $m2+=, 1
.endm

// Only a masked tail uses the explicit FP16 workspace (32 bytes per row).
// Complete panels never round-trip SRAM.
.macro SOFTMAX_BEGIN_TAIL
#ifdef ATTENTION_OUTPUT_F8
mov $m1, $m2
ld32 $m2, $mvertex_base, $m15, 6
shl $m0, $m4, 5
add $m2, $m2, $m0
#endif
.endm

.macro SOFTMAX_END_TAIL
#ifdef ATTENTION_OUTPUT_F8
add $m0, $m2, -32
mov $m2, $m1
.rept 2
ld64step $a0:1, $mzero, $m0+=, 1
ld64step $a2:3, $mzero, $m0+=, 1
f16v8tof8 $a0:1, $a0:3
st64step $a0:1, $mzero, $m2+=, 1
.endr
#endif
.endm
