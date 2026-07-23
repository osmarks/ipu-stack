#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class CheckFiniteF16 : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> input;
  Output<Vector<unsigned, VectorLayout::ONE_PTR>> flags;
  unsigned elements;

  bool compute(unsigned worker) {
    unsigned firstNonFinite = 0;
    const unsigned short *bits =
        reinterpret_cast<const unsigned short *>(&input[0]);
    for (unsigned index = worker; index < elements; index += 6) {
      if ((bits[index] & 0x7c00u) == 0x7c00u) {
        firstNonFinite = index + 1;
        break;
      }
    }
    flags[worker] = firstNonFinite;
    return true;
  }
};
