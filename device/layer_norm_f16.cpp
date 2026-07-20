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
      float mean = 0.0f;
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned column = 0; column < 16; ++column)
          mean += static_cast<float>(input[base + column]);
      }
      mean /= static_cast<float>(columns);

      float variance = 0.0f;
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned column = 0; column < 16; ++column) {
          const float centered = static_cast<float>(input[base + column]) - mean;
          variance += centered * centered;
        }
      }
      const float epsilon = static_cast<float>(epsilonQ30) * 0x1p-30f;
      const float scale = 1.0f / __builtin_sqrtf(
                                     variance / static_cast<float>(columns) +
                                     epsilon);
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
          const float2 scales = __builtin_convertvector(gammas, float2);
          const float2 biases = __builtin_convertvector(betas, float2);
          const float2 normalized = (values - mean) * scale * scales + biases;
          *reinterpret_cast<half2 *>(&output[base + inPanel]) =
              __builtin_convertvector(normalized, half2);
        }
      }
    }
    return true;
  }
};
