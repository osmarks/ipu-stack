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
#if INPUT_BYTES == 2 && OUTPUT_BYTES == 1
// Only boundary panels use guarded loads. Complete panels retain the pipelined
// loop below; entirely padded panels require no reads or floating-point work.
static __attribute__((noinline)) void castRowTail(
    const half *source, unsigned char *target, unsigned rows,
    unsigned sourceStride, unsigned stride, unsigned columns) {
  for (unsigned block = 0; block < 32; block += 8) {
    const half *input = source + block;
    unsigned char *output = target + block;
    unsigned src = 0, dst = 0;
    if (block + 8 <= columns) {
      asm volatile(
          "{ rpt %[rows], 4; fnop }\n"
          "{ ld64 $a0:1, %[in], %[src], 0; fnop }\n"
          "{ ld64 $a2:3, %[in], %[src], 1; fnop }\n"
          "{ add %[src], %[src], %[sourceStride]; f16v8tof8 $a4:5, $a0:3 }\n"
          "{ st64 $a4:5, %[out], %[dst], 0; fnop }\n"
          "{ add %[dst], %[dst], %[stride]; fnop }\n"
          : [src] "+&r"(src), [dst] "+&r"(dst)
          : [in] "r"(input), [out] "r"(output), [rows] "r"(rows),
            [stride] "r"(stride), [sourceStride] "r"(sourceStride)
          : "$a0:1", "$a2:3", "$a4:5", "memory");
    } else if (block < columns) {
      // Row strides are multiples of four halves; the only short vector
      // contains exactly four values and never reads past the row.
      asm volatile(
          "zero $a2:3\n"
          "{ rpt %[rows], 3; fnop }\n"
          "{ ld64 $a0:1, %[in], %[src], 0; fnop }\n"
          "{ add %[src], %[src], %[sourceStride]; f16v8tof8 $a4:5, $a0:3 }\n"
          "{ st64 $a4:5, %[out], %[dst], 0; fnop }\n"
          "{ add %[dst], %[dst], %[stride]; fnop }\n"
          : [src] "+&r"(src), [dst] "+&r"(dst)
          : [in] "r"(input), [out] "r"(output), [rows] "r"(rows),
            [stride] "r"(stride), [sourceStride] "r"(sourceStride)
          : "$a0:1", "$a2:3", "$a4:5", "memory");
    } else {
      asm volatile(
          "zero $a0:1\n"
          "{ rpt %[rows], 1; fnop }\n"
          "{ st64 $a0:1, %[out], %[dst], 0; fnop }\n"
          "{ add %[dst], %[dst], %[stride]; fnop }\n"
          : [dst] "+&r"(dst)
          : [out] "r"(output), [rows] "r"(rows), [stride] "r"(stride)
          : "$a0:1", "memory");
    }
  }
}

static __attribute__((noinline)) void zeroPackedRows(
    unsigned char *output, unsigned rows, unsigned stride) {
  unsigned offset = 0;
  asm volatile(
      "{ rpt %[rows], 4; fnop }\n"
      "{ st64 $azeros, %[out], %[offset], 0; fnop }\n"
      "{ st64 $azeros, %[out], %[offset], 1; fnop }\n"
      "{ st64 $azeros, %[out], %[offset], 2; fnop }\n"
      "{ st64 $azeros, %[out], %[offset], 3; fnop }\n"
      "{ add %[offset], %[offset], %[stride]; fnop }\n"
      : [offset] "+&r"(offset)
      : [out] "r"(output), [rows] "r"(rows), [stride] "r"(stride)
      : "memory");
}

// Keep the pipeline's register pressure off the small/fallback cast paths.
static __attribute__((noinline)) void castPackedRows(
    const half *first, const half *second, unsigned char *target,
    unsigned rows, unsigned stride, unsigned sourceStride) {
  // Pipeline both 16-value source streams independently. Each pair
  // of bundles stores one converted vector and loads the next one.
  // The final row drains the pipeline without speculative reads.
  const unsigned jump = sourceStride / 8 - 2;
  const unsigned steps = jump | (2 << 10) | ((stride / 8 - 1) << 20);
  const half *upperFirst = first;
  const half *upperSecond = second;
  asm volatile(
      "ld64step $a0:1, $mzero, %[upperFirst]+=, 1\n"
      "ld64step $a2:3, $mzero, %[upperFirst]+=, 2\n"
      "ld64step $a4:5, $mzero, %[upperSecond]+=, 1\n"
      "ld64step $a6:7, $mzero, %[upperSecond]+=, 2\n"
      "sub $m0, %[upperFirst], 8\n"
      "tapack $m0:1, $m0, $mzero, %[out]\n"
      "sub $m2, %[upperSecond], 8\n"
      "add $m3, %[out], 16\n"
      "tapack $m2:3, $m2, $mzero, $m3\n"
      "{ rpt %[rounds], 7; fnop }\n"
      "{ ld64step $a2:3, $mzero, %[upperFirst]+=, %[jump]; f16v8tof8 $a0:1, $a0:3 }\n"
      "{ ldst64pace $a0:1, $a0:1, $m0:1+=, %[steps], 1; fnop }\n"
      "{ ld64step $a2:3, $mzero, %[upperFirst]+=, 2; f16v8tof8 $a0:1, $a0:3 }\n"
      "{ ldst64pace $a0:1, $a0:1, $m0:1+=, %[steps], 14; fnop }\n"
      "{ ld64step $a6:7, $mzero, %[upperSecond]+=, %[jump]; f16v8tof8 $a4:5, $a4:7 }\n"
      "{ ldst64pace $a4:5, $a4:5, $m2:3+=, %[steps], 1; fnop }\n"
      "{ ld64step $a6:7, $mzero, %[upperSecond]+=, 2; f16v8tof8 $a4:5, $a4:7 }\n"
      "{ ldst64pace $a4:5, $a4:5, $m2:3+=, %[steps], 14; fnop }\n"
      // Upper pointers now address the last row's second vector.
      "{ ld64 $a2:3, %[upperFirst], $mzero, 0; f16v8tof8 $a0:1, $a0:3 }\n"
      "{ ldst64pace $a0:1, $a0:1, $m0:1+=, %[steps], 1; fnop }\n"
      "{ nop; f16v8tof8 $a2:3, $a0:3 }\n"
      "{ ld64 $a6:7, %[upperSecond], $mzero, 0; f16v8tof8 $a4:5, $a4:7 }\n"
      "{ ldst64pace $a4:5, $a4:5, $m2:3+=, %[steps], 1; fnop }\n"
      "{ st64 $a2:3, %[lastOut], $mzero, 1; f16v8tof8 $a6:7, $a4:7 }\n"
      "{ st64 $a6:7, %[lastOut], $mzero, 3; fnop }\n"
      : [upperFirst] "+&r"(upperFirst), [upperSecond] "+&r"(upperSecond)
      : [out] "r"(target),
        [lastOut] "r"(target + (rows - 1) * stride), [rounds] "r"(rows - 1),
        [jump] "r"(jump), [steps] "r"(steps)
      : "$m0", "$m1", "$m2", "$m3", "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
}
static __attribute__((always_inline)) inline void castFullRows(
    const half *first, const half *second, unsigned char *target,
    unsigned rows, unsigned stride, unsigned sourceStride) {
        asm volatile(
            "{ rpt %[rows], 11; fnop }\n"
            "{ ld64step $a0:1, $mzero, %[first]+=, 1; fnop }\n"
            "{ ld64step $a2:3, $mzero, %[first]+=, 1; fnop }\n"
            "{ ld64step $a0:1, $mzero, %[first]+=, 1; f16v8tof8 $a4:5, $a0:3 }\n"
            "{ ld64step $a2:3, $mzero, %[first]+=, %[srcJump]; fnop }\n"
            "{ st64step $a4:5, $mzero, %[out]+=, 1; f16v8tof8 $a6:7, $a0:3 }\n"
            "{ ld64step $a0:1, $mzero, %[second]+=, 1; fnop }\n"
            "{ ld64step $a2:3, $mzero, %[second]+=, 1; fnop }\n"
            "{ st64step $a6:7, $mzero, %[out]+=, 1; f16v8tof8 $a4:5, $a0:3 }\n"
            "{ ld64step $a0:1, $mzero, %[second]+=, 1; fnop }\n"
            "{ ld64step $a2:3, $mzero, %[second]+=, %[srcJump]; fnop }\n"
            "{ st64step $a4:5, $mzero, %[out]+=, 1; f16v8tof8 $a6:7, $a0:3 }\n"
            "{ st64step $a6:7, $mzero, %[out]+=, %[dstJump]; fnop }\n"
            : [first] "+&r"(first), [second] "+&r"(second), [out] "+&r"(target)
            : [rows] "r"(rows), [srcJump] "r"(sourceStride / 8 - 3),
              [dstJump] "r"(stride / 8 - 3)
            : "$a0:1", "$a2:3", "$a4:5", "$a6:7", "memory");
}

static __attribute__((noinline)) void castPaddedMatrices(
    const half *source, unsigned char *destination, unsigned elements,
    unsigned panelRows, unsigned validColumns, unsigned sourceColumns,
    unsigned rowShape, unsigned worker) {
  const unsigned matrixRows = rowShape >> 16;
  const unsigned validRows = rowShape & 65535;
  const bool wholePanels = panelRows <= 32 && elements >= panelRows * 32 * 6;
  const unsigned stride = wholePanels ? 32 : 192;
  const unsigned sourceStride = sourceColumns * 2 * (wholePanels ? 1 : 6);
  const unsigned inStart = reinterpret_cast<unsigned>(source);
  const unsigned outStart = reinterpret_cast<unsigned>(destination);
  const bool separate = (((inStart + panelRows * sourceColumns * 2 - 1) >> 15) < (outStart >> 15) ||
      ((outStart + elements - 1) >> 15) < (inStart >> 15));
  for (unsigned matrix = 0; matrix < panelRows; matrix += matrixRows) {
    const unsigned firstRow = wholePanels ? 0 : (worker + 6 - matrix % 6) % 6;
    if (firstRow >= matrixRows) continue;
    const unsigned row = matrix + firstRow;
    const unsigned physicalRows = wholePanels ? matrixRows : (matrixRows + 5 - firstRow) / 6;
    const unsigned rows = firstRow >= validRows ? 0 : wholePanels ? validRows : (validRows + 5 - firstRow) / 6;
    for (unsigned panel = wholePanels ? worker * panelRows * 32 : 0,
                  column = wholePanels ? worker * 32 : 0;
         panel < elements; panel += panelRows * 32 * (wholePanels ? 6 : 1),
                           column += wholePanels ? 192 : 32) {
      unsigned char *target = destination + panel + row * 32;
      if (physicalRows > rows)
        zeroPackedRows(target + rows * stride, physicalRows - rows, stride);
      if (!rows) continue;
      if (column >= validColumns) {
        zeroPackedRows(target, rows, stride);
        continue;
      }
      const half *first = source + row * sourceColumns + column;
      if (column + 32 > validColumns) {
        castRowTail(first, target, rows, sourceStride, stride, validColumns - column);
      } else if (separate && rows >= 16 && sourceStride / 8 - 2 < 512) {
        castPackedRows(first, first + 16, target, rows, stride, sourceStride);
      } else {
        castFullRows(first, first + 16, target, rows, stride, sourceStride);
      }
    }
  }
}
#endif
// Both worker entry points consume this one descriptor layout. For FP16
// packing, sourceMetadata holds row bounds and sourceExtent readable columns.
#define CAST_FIELDS \
  Input<Vector<Source, VectorLayout::ONE_PTR>> source; \
  Output<Vector<Destination, VectorLayout::ONE_PTR>> destination; \
  unsigned elements; \
  int sourceMetadata; \
  int destinationScale; \
  unsigned panelRows; \
  unsigned sourceExtent; \
  unsigned rowMajorColumns;

class CAST_VERTEX : public MultiVertex {
public:
  CAST_FIELDS
  bool compute(unsigned worker) {
    // Assembly writes cannot change the immutable call descriptor. Keep its
    // fields in registers rather than reloading them after each memory clobber.
    const Source *source = &this->source[0];
    Destination *destination = &this->destination[0];
    const unsigned elements = this->elements;
    const unsigned panelRows = this->panelRows;
    const unsigned sourceExtent = this->sourceExtent;
    const unsigned rowMajorColumns = this->rowMajorColumns;
    const unsigned sourceElements = rowMajorColumns
        ? (panelRows ? panelRows : 1) * rowMajorColumns : sourceExtent;
    const unsigned validColumns = sourceExtent;
#if INPUT_BYTES == 2 && OUTPUT_BYTES == 1
    setQuarterConfig({quarter_metadata::f143, static_cast<signed char>(-destinationScale)});
    const unsigned rowShape = rowMajorColumns ? static_cast<unsigned>(sourceMetadata) : 0;
    // Combined loads/stores require different memory elements. Group pairs
    // of standard banks conservatively so this also covers interleaved SRAM.
    const unsigned inStart = reinterpret_cast<unsigned>(source);
    const unsigned outStart = reinterpret_cast<unsigned>(destination);
    const bool separate = sourceElements && elements &&
        (((inStart + sourceElements * 2 - 1) >> 15) < (outStart >> 15) ||
         ((outStart + elements - 1) >> 15) < (inStart >> 15));
    if (panelRows) {
      const unsigned panelElements = panelRows * 32;
      // Small panels have too few rows to amortize six-worker setup. Give
      // workers complete panels when there are enough independent panels.
      const bool wholePanels = panelRows <= 32 && elements >= panelElements * 6;
      const unsigned row = wholePanels ? 0 : worker;
      const unsigned physicalRows = wholePanels ? panelRows : (panelRows + 5 - worker) / 6;
      const unsigned validRows = rowShape && (rowShape >> 16) == panelRows ? rowShape & 65535 : panelRows;
      const unsigned rows = row >= validRows ? 0 : wholePanels ? validRows : (validRows + 5 - worker) / 6;
      const unsigned stride = wholePanels ? 32 : 192;
      const unsigned sourceStride = rowMajorColumns ? rowMajorColumns * 2 * (wholePanels ? 1 : 6) : stride;
      const unsigned panelStep = panelElements * (wholePanels ? 6 : 1);
      unsigned column = wholePanels ? worker * 32 : 0;
      for (unsigned panel = wholePanels ? worker * panelElements : 0;
           panel < elements && row < panelRows; panel += panelStep, column += wholePanels ? 192 : 32) {
        unsigned char *panelTarget = &destination[panel + row * 32];
        if (physicalRows > rows)
          zeroPackedRows(panelTarget + rows * stride, physicalRows - rows, stride);
        if (!rows) continue;
        if (rowMajorColumns && column >= validColumns) {
          zeroPackedRows(panelTarget, rows, stride);
          continue;
        }
        const half *first = &source[rowMajorColumns ? row * rowMajorColumns + column : panel + row * 16];
        if (rowMajorColumns && column + 32 > validColumns) {
          castRowTail(first, panelTarget, rows, sourceStride, stride, validColumns - column);
          continue;
        }
        const half *second = first + (rowMajorColumns ? 16 : panelRows * 16);
        unsigned char *target = &destination[panel + row * 32];
        unsigned sourceOffset = 0, destinationOffset = 0;
        if (!rowMajorColumns && panel + panelElements > sourceElements) {
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
        // Amortize the pipeline call/setup, and fit the signed 10-bit strides.
        if (separate && rows >= 16 && sourceStride / 8 - 2 < 512) {
          castPackedRows(first, second, target, rows, stride, sourceStride);
          continue;
        }
        castFullRows(first, second, target, rows, stride, sourceStride);
      }
      return true;
    }
    const unsigned vectors = elements / 8;
    unsigned rounds = (vectors + 5 - worker) / 6;
    const unsigned workerRounds = rounds;
    const half *input = &source[worker * 8];
    unsigned char *output = &destination[worker * 8];
    if (separate && rounds >= 4) {
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
        static_cast<signed char>(INPUT_BYTES == 1 && OUTPUT_BYTES == 2 ? sourceMetadata : 0)});
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
        float element = static_cast<float>(value[lane]) * powerOfTwo(sourceMetadata - destinationScale);
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
        destination[base + lane] = static_cast<float>(value[lane]) * powerOfTwo(sourceMetadata);
#else
        destination[base + lane] = static_cast<Destination>(value[lane]);
#endif
#endif
    }
    return true;
#endif
  }
};

#if INPUT_BYTES == 2 && OUTPUT_BYTES == 1
// The supervisor selects this entry only for padding between batch matrices.
// Sharing the descriptor preserves one cast ABI without burdening the hot loop.
class Cast2To1Padded : public MultiVertex {
public:
  CAST_FIELDS
  bool compute(unsigned worker) {
    setQuarterConfig({quarter_metadata::f143, static_cast<signed char>(-destinationScale)});
    castPaddedMatrices(&source[0], &destination[0], elements, panelRows,
                       sourceExtent, rowMajorColumns,
                       static_cast<unsigned>(sourceMetadata), worker);
    return true;
  }
};
#endif

#undef CAST_FIELDS
