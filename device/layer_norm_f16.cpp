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
      float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
      float4 squareSums = {0.0f, 0.0f, 0.0f, 0.0f};
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned column = 0; column < 16; column += 4) {
          const half4 packed =
              *reinterpret_cast<const half4 *>(&input[base + column]);
          const float4 values = __builtin_convertvector(packed, float4);
          sums += values;
          squareSums += values * values;
        }
      }
      const float reciprocalColumns = 1.0f / static_cast<float>(columns);
      const float mean = (sums[0] + sums[1] + sums[2] + sums[3]) *
                         reciprocalColumns;
      const float secondMoment =
          (squareSums[0] + squareSums[1] + squareSums[2] + squareSums[3]) *
          reciprocalColumns;
      const float variance = __builtin_fmaxf(0.0f, secondMoment - mean * mean);
      const float epsilon = static_cast<float>(epsilonQ30) * 0x1p-30f;
      const float scale = 1.0f / __builtin_sqrtf(variance + epsilon);
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned inPanel = 0; inPanel < 16; inPanel += 4) {
          const unsigned column = panel * 16 + inPanel;
          const half4 inputs =
              *reinterpret_cast<const half4 *>(&input[base + inPanel]);
          const half4 gammas =
              *reinterpret_cast<const half4 *>(&affine[column]);
          const half4 betas =
              *reinterpret_cast<const half4 *>(&affine[columns + column]);
          const float4 values = __builtin_convertvector(inputs, float4);
          const float4 scales = __builtin_convertvector(gammas, float4);
          const float4 biases = __builtin_convertvector(betas, float4);
          const float4 normalized = (values - mean) * scale * scales + biases;
          *reinterpret_cast<half4 *>(&output[base + inPanel]) =
              __builtin_convertvector(normalized, half4);
        }
      }
    }
    return true;
  }
};
