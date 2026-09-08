#include <poplar/HalfFloat.hpp>
#include <poplar/QuarterFloat.hpp>
#include <poplar/Vertex.hpp>
using namespace poplar;

static float powerOfTwo(int exponent) {
  union { unsigned bits; float value; } scale;
  scale.bits = static_cast<unsigned>(127 + exponent) << 23;
  return scale.value;
}
#if INPUT_BYTES == 1
using Source = unsigned char;
#elif INPUT_BYTES == 2
using Source = half;
#else
using Source = float;
#endif
#if OUTPUT_BYTES == 1
using Destination = unsigned char;
#elif OUTPUT_BYTES == 2
using Destination = half;
#else
using Destination = float;
#endif
class CAST_VERTEX : public MultiVertex {
public:
  Input<Vector<Source, VectorLayout::ONE_PTR>> source;
  Output<Vector<Destination, VectorLayout::ONE_PTR>> destination;
  unsigned elements;
  int sourceScale;
  int destinationScale;
  unsigned panelRows;
  unsigned sourceElements;
  unsigned rowMajorColumns;
  bool compute(unsigned worker) {
    // Assembly writes cannot change the immutable call descriptor. Keep its
    // fields in registers rather than reloading them after each memory clobber.
    const Source *source = &this->source[0];
    Destination *destination = &this->destination[0];
    const unsigned elements = this->elements;
    const unsigned panelRows = this->panelRows;
    const unsigned sourceElements = this->sourceElements;
    const unsigned rowMajorColumns = this->rowMajorColumns;
#if INPUT_BYTES == 2 && OUTPUT_BYTES == 1
    setQuarterConfig({quarter_metadata::f143, static_cast<signed char>(-destinationScale)});
    if (panelRows) {
      const unsigned panelElements = panelRows * 32;
      // Small panels have too few rows to amortize six-worker setup. Give
      // workers complete panels when there are enough independent panels.
      const bool wholePanels = panelRows <= 32 && elements >= panelElements * 6;
      const unsigned row = wholePanels ? 0 : worker;
      const unsigned rows = wholePanels ? panelRows : (panelRows + 5 - worker) / 6;
      const unsigned stride = wholePanels ? 32 : 192;
      const unsigned sourceStride = rowMajorColumns ? rowMajorColumns * 2 * (wholePanels ? 1 : 6) : stride;
      const unsigned panelStep = panelElements * (wholePanels ? 6 : 1);
      unsigned column = wholePanels ? worker * 32 : 0;
      for (unsigned panel = wholePanels ? worker * panelElements : 0;
           panel < elements && row < panelRows; panel += panelStep, column += wholePanels ? 192 : 32) {
        const half *first = &source[rowMajorColumns ? row * rowMajorColumns + column : panel + row * 16];
        const half *second = first + (rowMajorColumns ? 16 : panelRows * 16);
        unsigned char *target = &destination[panel + row * 32];
        unsigned sourceOffset = 0, destinationOffset = 0;
        if (panel + panelElements > sourceElements) {
          // A producer may own only a 16-element tail. Populate the other
          // half of the FP8 panel here, without a padded F16 staging copy.
          asm volatile(
              "{ rpt %[rows], 9; fnop }\n"
              "{ ld64 $a0:1, %[first], %[src], 0; fnop }\n"
              "{ ld64 $a2:3, %[first], %[src], 1; fnop }\n"
              "{ ld64 $a0:1, %[first], %[src], 2; f16v8tof8 $a4:5, $a0:3 }\n"
              "{ ld64 $a2:3, %[first], %[src], 3; fnop }\n"
              "{ st64 $a4:5, %[out], %[dst], 0; f16v8tof8 $a6:7, $a0:3 }\n"
              "{ st64 $a6:7, %[out], %[dst], 1; zero $a4:5 }\n"
              "{ st64 $a4:5, %[out], %[dst], 2; fnop }\n"
              "{ st64 $a4:5, %[out], %[dst], 3; fnop }\n"
              "{ add %[src], %[src], %[srcStride]; fnop }\n"
              "{ add %[dst], %[dst], %[stride]; fnop }\n"
              : [src] "+&r"(sourceOffset), [dst] "+&r"(destinationOffset)
              : [first] "r"(first), [out] "r"(target), [rows] "r"(rows), [stride] "r"(stride), [srcStride] "r"(sourceStride)
              : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
          continue;
        }
        // Each repeat converts 32 values, without a software row loop.
        // Offsets advance to this worker's next row. Packing
        // needs two strided source panels and one contiguous destination row.
        asm volatile(
            "{ rpt %[rows], 13; fnop }\n"
            "{ ld64 $a0:1, %[first], %[src], 0; fnop }\n"
            "{ ld64 $a2:3, %[first], %[src], 1; fnop }\n"
            "{ ld64 $a0:1, %[first], %[src], 2; f16v8tof8 $a4:5, $a0:3 }\n"
            "{ ld64 $a2:3, %[first], %[src], 3; fnop }\n"
            "{ st64 $a4:5, %[out], %[dst], 0; f16v8tof8 $a6:7, $a0:3 }\n"
            "{ ld64 $a0:1, %[second], %[src], 0; fnop }\n"
            "{ ld64 $a2:3, %[second], %[src], 1; fnop }\n"
            "{ st64 $a6:7, %[out], %[dst], 1; f16v8tof8 $a4:5, $a0:3 }\n"
            "{ ld64 $a0:1, %[second], %[src], 2; fnop }\n"
            "{ ld64 $a2:3, %[second], %[src], 3; fnop }\n"
            "{ st64 $a4:5, %[out], %[dst], 2; f16v8tof8 $a6:7, $a0:3 }\n"
            "{ st64 $a6:7, %[out], %[dst], 3; fnop }\n"
            "{ add %[src], %[src], %[srcStride]; fnop }\n"
            "{ add %[dst], %[dst], %[stride]; fnop }\n"
            : [src] "+&r"(sourceOffset), [dst] "+&r"(destinationOffset)
            : [first] "r"(first), [second] "r"(second), [out] "r"(target), [rows] "r"(rows), [stride] "r"(stride), [srcStride] "r"(sourceStride)
            : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
      }
      return true;
    }
    const unsigned vectors = elements / 8;
    unsigned rounds = (vectors + 5 - worker) / 6;
    const unsigned workerRounds = rounds;
    const half *input = &source[worker * 8];
    unsigned char *output = &destination[worker * 8];
    // Conservatively treat each 32 KiB address group as one memory element.
    // This covers both 16 KiB standard banks and two-bank interleaved groups,
    // including Repeat's runtime pointers, without constraining placement.
    const unsigned inStart = reinterpret_cast<unsigned>(&source[0]);
    const unsigned outStart = reinterpret_cast<unsigned>(&destination[0]);
    const bool separate = rounds >= 4 && sourceElements && elements &&
        (((inStart + sourceElements * 2 - 1) >> 15) < (outStart >> 15) ||
         ((outStart + elements - 1) >> 15) < (inStart >> 15));
    if (separate) {
      // Pipeline one conversion ahead, as in the SDK half-to-quarter loop.
      // The combined load/store reads the next lower half while storing the
      // previous result. Prologue/epilogue avoid reading past the last vector.
      --rounds;
      asm volatile(
          "ld64step $a0:1, $mzero, %[in]+=, 1\n"
          "ld64step $a2:3, $mzero, %[in]+=, 11\n"
          "tapack $m0:1, %[in], $mzero, %[out]\n"
          "ld64step $azeros, $mzero, %[in]+=, 1\n"
          "mul $m2, %[rounds], 48\n"
          "add %[out], %[out], $m2\n"
          "setzi $m2, (12<<10)|6\n"
          "{ rpt %[rounds], 1; fnop }\n"
          "{ ld64step $a2:3, $mzero, %[in]+=, 12; f16v8tof8 $a0:1, $a0:3 }\n"
          "{ ldst64pace $a0:1, $a0:1, $m0:1+=, $m2, 6; fnop }\n"
          "f16v8tof8 $a2:3, $a0:3\n"
          "st64step $a2:3, $mzero, %[out]+=, 6\n"
          : [in] "+&r"(input), [out] "+&r"(output)
          : [rounds] "r"(rounds)
          : "$m0", "$m1", "$m2", "$a0:1", "$a2:3", "memory");
    } else {
      asm volatile(
          "{ rpt %[rounds], 3; fnop }\n"
          "{ ld64step $a0:1, $mzero, %[in]+=, 1; fnop }\n"
          "{ ld64step $a2:3, $mzero, %[in]+=, 11; fnop }\n"
          "{ nop; f16v8tof8 $a4:5, $a0:3 }\n"
          "{ st64step $a4:5, $mzero, %[out]+=, 6; fnop }\n"
          : [in] "+&r"(input), [out] "+&r"(output) : [rounds] "r"(rounds)
          : "$a0:1", "$a2:3", "$a4:5", "memory");
    }
    const unsigned base = worker * 8 + workerRounds * 48;
    if (base < elements) {
      half4 lower = {half(0), half(0), half(0), half(0)};
      half4 upper = lower;
      for (unsigned lane = 0; lane < 4; ++lane) {
        if (base + lane < elements) lower[lane] = source[base + lane];
        if (base + lane + 4 < elements) upper[lane] = source[base + lane + 4];
      }
      union { float2 packed; unsigned words[2]; } output;
      asm volatile("mov $a0:1, %[lo]\nmov $a2:3, %[hi]\nf16v8tof8 %[out], $a0:3"
          : [out] "=r"(output.packed) : [lo] "r"(lower), [hi] "r"(upper) : "$a0:1", "$a2:3");
      for (unsigned word = 0; word < 2 && base + word * 4 < elements; ++word) {
        volatile unsigned *target = reinterpret_cast<volatile unsigned *>(&destination[base + word * 4]);
        const unsigned remaining = elements - base - word * 4;
        if (remaining >= 4) *target = output.words[word];
        else {
          const unsigned mask = (1u << (remaining * 8)) - 1;
          *target = (*target & ~mask) | (output.words[word] & mask);
        }
      }
    }
    return true;
#else
    setQuarterConfig({quarter_metadata::f143,
        static_cast<signed char>(INPUT_BYTES == 1 && OUTPUT_BYTES == 2 ? sourceScale : 0)});
    // A worker owns complete destination words, including the final RMW.
    for (unsigned base = worker * 4; base < elements; base += 24) {
      half4 value;
#if INPUT_BYTES == 1
      // Decode to an unscaled half first when the destination has greater
      // range, or a different FP8 scale. All finite F143 codes fit in half.
      union { unsigned word; float packed; } input;
      input.word = *reinterpret_cast<const unsigned *>(&source[base]);
      asm volatile("f8v4tof16 %[out], %[in]" : [out] "=r"(value) : [in] "r"(input.packed));
#else
      for (unsigned lane = 0; lane < 4; ++lane)
        {
          float element = base + lane < elements ? static_cast<float>(source[base + lane]) : 0.0f;
#if INPUT_BYTES == 4 && OUTPUT_BYTES == 1
          // Scaling after an intermediate half conversion would overflow on
          // valid, positively scaled FP8 values outside the half range.
          element *= powerOfTwo(-destinationScale);
          element = element > 240.0f ? 240.0f : element;
          element = element < -240.0f ? -240.0f : element;
#endif
          value[lane] = static_cast<half>(element);
        }
#endif
#if OUTPUT_BYTES == 1
#if INPUT_BYTES == 1
      for (unsigned lane = 0; lane < 4; ++lane) {
        float element = static_cast<float>(value[lane]) * powerOfTwo(sourceScale - destinationScale);
        element = element > 240.0f ? 240.0f : element;
        element = element < -240.0f ? -240.0f : element;
        value[lane] = static_cast<half>(element);
      }
#endif
      const half4 zero = {half(0), half(0), half(0), half(0)};
      union { float2 packed; unsigned words[2]; } output;
      asm volatile("mov $a0:1, %[in]\nmov $a2:3, %[zero]\nf16v8tof8 %[out], $a0:3"
          : [out] "=r"(output.packed) : [in] "r"(value), [zero] "r"(zero) : "$a0:1", "$a2:3");
      volatile unsigned *words = reinterpret_cast<volatile unsigned *>(&destination[base]);
      const unsigned remaining = elements - base;
      if (remaining >= 4) *words = output.words[0];
      else {
        const unsigned mask = (1u << (8 * remaining)) - 1;
        *words = (*words & ~mask) | (output.words[0] & mask);
      }
#else
      for (unsigned lane = 0; lane < 4 && base + lane < elements; ++lane)
#if INPUT_BYTES == 1 && OUTPUT_BYTES == 4
        destination[base + lane] = static_cast<float>(value[lane]) * powerOfTwo(sourceScale);
#else
        destination[base + lane] = static_cast<Destination>(value[lane]);
#endif
#endif
    }
    return true;
#endif
  }
};
