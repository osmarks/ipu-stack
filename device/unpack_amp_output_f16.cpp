#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef UNPACK_COLUMNS
#define UNPACK_COLUMNS 16
#endif
#ifndef UNPACK_VERTEX_NAME
#define UNPACK_VERTEX_NAME RearrangeAmpOutputToRowMajorF16
#endif

static_assert(UNPACK_COLUMNS % 16 == 0);

class UNPACK_VERTEX_NAME : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned rows;

  bool compute(unsigned worker) {
    const unsigned *sourceWords = reinterpret_cast<const unsigned *>(&source[0]);
    unsigned *destinationWords = reinterpret_cast<unsigned *>(&destination[0]);
    const unsigned words = rows * UNPACK_COLUMNS / 2;
    for (unsigned word = worker; word < words; word += 6) {
      const unsigned row = word * 2 / UNPACK_COLUMNS;
      const unsigned column = word * 2 % UNPACK_COLUMNS;
      unsigned packed = 0;
      for (unsigned halfIndex = 0; halfIndex != 2; ++halfIndex) {
        const unsigned logicalColumn = column + halfIndex;
        const unsigned logicalPair = logicalColumn % 16 / 2;
        const unsigned physicalPair = logicalPair % 4 * 2 + logicalPair / 4;
        const unsigned physicalColumn = physicalPair * 2 + logicalColumn % 2;
        const unsigned sourceElement = logicalColumn / 16 * rows * 16 +
                                       row * 16 + physicalColumn;
        const unsigned value =
            (sourceWords[sourceElement / 2] >> ((sourceElement & 1) * 16)) & 0xffff;
        packed |= value << (halfIndex * 16);
      }
      destinationWords[word] = packed;
    }
    return true;
  }
};
