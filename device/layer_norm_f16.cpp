#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class LayerNormAffineF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> input;
  Input<Vector<half, VectorLayout::ONE_PTR>> affine;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned columns;
  unsigned epsilonQ30;

  bool compute(unsigned worker) {
    for (unsigned row = worker; row < rows; row += 6) {
      float sum = 0.0f;
      float squareSum = 0.0f;
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned column = 0; column < 16; column += 2) {
          const half2 packed =
              *reinterpret_cast<const half2 *>(&input[base + column]);
          const float2 values = __builtin_convertvector(packed, float2);
          sum += values[0] + values[1];
          squareSum += values[0] * values[0] + values[1] * values[1];
        }
      }
      const float reciprocalColumns = 1.0f / static_cast<float>(columns);
      const float mean = sum * reciprocalColumns;
      const float secondMoment = squareSum * reciprocalColumns;
      const float variance = __builtin_fmaxf(0.0f, secondMoment - mean * mean);
      const float epsilon = static_cast<float>(epsilonQ30) * 0x1p-30f;
      const float scale = 1.0f / __builtin_sqrtf(variance + epsilon);
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned inPanel = 0; inPanel < 16; inPanel += 2) {
          const unsigned column = panel * 16 + inPanel;
          const half2 inputs =
              *reinterpret_cast<const half2 *>(&input[base + inPanel]);
          const half2 gammas =
              *reinterpret_cast<const half2 *>(&affine[column]);
          const half2 betas =
              *reinterpret_cast<const half2 *>(&affine[columns + column]);
          const float2 values = __builtin_convertvector(inputs, float2);
          const float2 gammasF32 = __builtin_convertvector(gammas, float2);
          const float2 betasF32 = __builtin_convertvector(betas, float2);
          const float2 normalized =
              (values - mean) * scale * gammasF32 + betasF32;
          *reinterpret_cast<half2 *>(&output[base + inPanel]) =
              __builtin_convertvector(normalized, half2);
        }
      }
    }
    return true;
  }
};
