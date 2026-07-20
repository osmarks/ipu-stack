#include <poplar/HalfFloat.hpp>
#include <poplar/QuarterFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class ExpandF8F143ToF16 : public MultiVertex {
public:
  Input<Vector<unsigned char, VectorLayout::ONE_PTR>> input;
  Input<Vector<unsigned char, VectorLayout::ONE_PTR>> unused;
  Output<Vector<half, VectorLayout::ONE_PTR>> output;
  unsigned elements;
  int scale;

  bool compute(unsigned worker) {
    const quarter_metadata metadata = {quarter_metadata::f143,
                                       static_cast<signed char>(scale)};
    setQuarterConfig(metadata);
    const unsigned groups = elements / 4;
    for (unsigned group = worker; group < groups; group += 6) {
      const float packed =
          *reinterpret_cast<const float *>(&input[group * 4]);
      half4 expanded;
      asm volatile("f8v4tof16 %[expanded], %[packed]"
                   : [expanded] "=r"(expanded)
                   : [packed] "r"(packed));
      *reinterpret_cast<half4 *>(&output[group * 4]) = expanded;
    }
    return true;
  }
};
