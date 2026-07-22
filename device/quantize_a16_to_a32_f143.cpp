#include <poplar/HalfFloat.hpp>
#include <poplar/QuarterFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class QuantizeA16ToA32F143 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> input;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<unsigned char, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned columns;
  int scale;

  bool compute(unsigned worker) {
    const auto outputScale = static_cast<signed char>(scale);
    const quarter_metadata metadata = {
        quarter_metadata::f143,
        static_cast<signed char>(-outputScale),
    };
    setQuarterConfig(metadata);

    for (unsigned row = worker; row < rows; row += 6) {
      for (unsigned group = 0; group < columns / 32; ++group) {
        const unsigned outputBase = (group * rows + row) * 32;
        for (unsigned panel = 0; panel < 2; ++panel) {
          const unsigned inputBase = ((group * 2 + panel) * rows + row) * 16;
          for (unsigned halfGroup = 0; halfGroup < 2; ++halfGroup) {
            const half4 lower = *reinterpret_cast<const half4 *>(
                &input[inputBase + halfGroup * 8]);
            const half4 upper = *reinterpret_cast<const half4 *>(
                &input[inputBase + halfGroup * 8 + 4]);
            float2 packed;
            asm volatile("mov $a0:1, %[lower]\n"
                         "mov $a2:3, %[upper]\n"
                         "f16v8tof8 %[packed], $a0:3"
                         : [packed] "=r"(packed)
                         : [lower] "r"(lower), [upper] "r"(upper)
                         : "$a0:1", "$a2:3");
            *reinterpret_cast<float2 *>(
                &output[outputBase + panel * 16 + halfGroup * 8]) = packed;
          }
        }
      }
    }
    return true;
  }
};

class QuantizeReblockA16ToA32F143 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> input;
  Input<Vector<half, VectorLayout::ONE_PTR>> unused;
  Output<Vector<unsigned char, VectorLayout::ONE_PTR>> output;
  unsigned dimensions;
  unsigned offsets;
  unsigned countAndScale;

  bool compute(unsigned worker) {
    const unsigned sourceRows = dimensions & 0x3ff;
    const unsigned destinationRows = dimensions >> 10;
    const unsigned sourceRowStart = offsets & 0x3ff;
    const unsigned destinationRowStart = offsets >> 10;
    const unsigned copyRows = countAndScale & 0x3ff;
    const unsigned scaleBits = (countAndScale >> 10) & 0x3f;
    const unsigned groupCount = countAndScale >> 16;
    const auto outputScale =
        static_cast<signed char>((scaleBits ^ 0x20) - 0x20);
    const quarter_metadata metadata = {
        quarter_metadata::f143,
        static_cast<signed char>(-outputScale),
    };
    setQuarterConfig(metadata);

    for (unsigned copiedRow = worker; copiedRow < copyRows; copiedRow += 6) {
      const unsigned sourceRow = sourceRowStart + copiedRow;
      const unsigned destinationRow = destinationRowStart + copiedRow;
      for (unsigned group = 0; group < groupCount; ++group) {
        const unsigned outputBase = (group * destinationRows + destinationRow) * 32;
        for (unsigned panel = 0; panel < 2; ++panel) {
          const unsigned inputBase =
              ((group * 2 + panel) * sourceRows + sourceRow) * 16;
          for (unsigned halfGroup = 0; halfGroup < 2; ++halfGroup) {
            const half4 lower = *reinterpret_cast<const half4 *>(
                &input[inputBase + halfGroup * 8]);
            const half4 upper = *reinterpret_cast<const half4 *>(
                &input[inputBase + halfGroup * 8 + 4]);
            float2 packed;
            asm volatile("mov $a0:1, %[lower]\n"
                         "mov $a2:3, %[upper]\n"
                         "f16v8tof8 %[packed], $a0:3"
                         : [packed] "=r"(packed)
                         : [lower] "r"(lower), [upper] "r"(upper)
                         : "$a0:1", "$a2:3");
            *reinterpret_cast<float2 *>(
                &output[outputBase + panel * 16 + halfGroup * 8]) = packed;
          }
        }
      }
    }
    return true;
  }
};
