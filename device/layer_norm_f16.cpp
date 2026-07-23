#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

template <bool AddResidual, typename InputVector, typename RightVector,
          typename ResidualVector, typename AffineVector, typename OutputVector>
static __attribute__((always_inline)) bool
normalize(const InputVector &input, const RightVector &right,
          ResidualVector &residual, const AffineVector &affine,
          OutputVector &output, unsigned rows, unsigned columns,
          unsigned epsilonQ30, unsigned worker) {
    for (unsigned row = worker; row < rows; row += 6) {
      float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};
      float4 squareSums = {0.0f, 0.0f, 0.0f, 0.0f};
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned column = 0; column < 16; column += 4) {
          half4 packed =
              *reinterpret_cast<const half4 *>(&input[base + column]);
          if (AddResidual) {
            packed +=
                *reinterpret_cast<const half4 *>(&right[base + column]);
            *reinterpret_cast<half4 *>(&residual[base + column]) = packed;
          }
          const float4 values = __builtin_convertvector(packed, float4);
          sums += values;
          squareSums += values * values;
        }
      }
      const float sum = sums[0] + sums[1] + sums[2] + sums[3];
      const float squareSum =
          squareSums[0] + squareSums[1] + squareSums[2] + squareSums[3];
      const float reciprocalColumns = 1.0f / static_cast<float>(columns);
      const float mean = sum * reciprocalColumns;
      const float secondMoment = squareSum * reciprocalColumns;
      const float variance = __builtin_fmaxf(0.0f, secondMoment - mean * mean);
      const float epsilon = static_cast<float>(epsilonQ30) * 0x1p-30f;
      const float scale = 1.0f / __builtin_sqrtf(variance + epsilon);
      const half4 meanF16 =
          __builtin_convertvector((float4){mean, mean, mean, mean}, half4);
      const half4 scaleF16 =
          __builtin_convertvector((float4){scale, scale, scale, scale}, half4);
      for (unsigned panel = 0; panel < columns / 16; ++panel) {
        const unsigned base = panel * rows * 16 + row * 16;
        for (unsigned inPanel = 0; inPanel < 16; inPanel += 4) {
          const unsigned column = panel * 16 + inPanel;
          const half4 inputs = *reinterpret_cast<const half4 *>(
              AddResidual ? &residual[base + inPanel]
                          : &input[base + inPanel]);
          const half4 gammas =
              *reinterpret_cast<const half4 *>(&affine[column]);
          const half4 betas =
              *reinterpret_cast<const half4 *>(&affine[columns + column]);
          *reinterpret_cast<half4 *>(&output[base + inPanel]) =
              (inputs - meanF16) * scaleF16 * gammas + betas;
        }
      }
    }
    return true;
}

class LayerNormAffineF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> input;
  Input<Vector<half, VectorLayout::ONE_PTR>> affine;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned columns;
  unsigned epsilonQ30;

  bool compute(unsigned worker) {
    return normalize<false>(input, input, output, affine, output, rows, columns,
                            epsilonQ30, worker);
  }
};

class AddLayerNormAffineF16 : public MultiVertex {
public:
  InOut<Vector<half, VectorLayout::ONE_PTR>> residual;
  Input<Vector<half, VectorLayout::ONE_PTR>> right;
  Input<Vector<half, VectorLayout::ONE_PTR>> affine;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned rows;
  unsigned columns;
  unsigned epsilonQ30;

  bool compute(unsigned worker) {
    return normalize<true>(residual, right, residual, affine, output, rows,
                           columns, epsilonQ30, worker);
  }
};
