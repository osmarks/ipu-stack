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
  bool compute(unsigned worker) {
#if INPUT_BYTES == 2 && OUTPUT_BYTES == 1
    setQuarterConfig({quarter_metadata::f143, static_cast<signed char>(-destinationScale)});
    unsigned base = worker * 8;
    for (; base + 8 <= elements; base += 48) {
      const half4 lower = *reinterpret_cast<const half4 *>(&source[base]);
      const half4 upper = *reinterpret_cast<const half4 *>(&source[base + 4]);
      float2 packed;
      asm volatile("mov $a0:1, %[lo]\nmov $a2:3, %[hi]\nf16v8tof8 %[out], $a0:3"
          : [out] "=r"(packed) : [lo] "r"(lower), [hi] "r"(upper) : "$a0:1", "$a2:3");
      *reinterpret_cast<float2 *>(&destination[base]) = packed;
    }
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
