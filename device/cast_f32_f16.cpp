#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

class CastF32ToF16 : public MultiVertex {
public:
  Input<Vector<float, VectorLayout::ONE_PTR>> source;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned elements;

  bool compute(unsigned worker) {
    // Assign complete destination words to workers: separate halfword stores
    // from different workers would race on the same SRAM word.
    for (unsigned pair = worker; pair < elements / 2; pair += 6) {
      const float2 value = *reinterpret_cast<const float2 *>(&source[2 * pair]);
      *reinterpret_cast<half2 *>(&destination[2 * pair]) =
          __builtin_convertvector(value, half2);
    }
    if (worker == 0 && (elements & 1)) {
      union { half2 lanes; unsigned word; } tail;
      const float2 value = {source[elements - 1], 0.0f};
      tail.lanes = __builtin_convertvector(value, half2);
      // Keep a real word RMW; otherwise popc folds it into a __st16 call.
      volatile unsigned *words =
          reinterpret_cast<volatile unsigned *>(&destination[0]);
      words[elements / 2] = (words[elements / 2] & 0xffff0000u) |
                           (tail.word & 0xffffu);
    }
    return true;
  }
};
