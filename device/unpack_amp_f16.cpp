#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef UNPACK_SOURCE_ORDER
#define UNPACK_SOURCE_ORDER 0
#endif
#ifndef UNPACK_LOGICAL_ROWS
#define UNPACK_LOGICAL_ROWS 16
#endif
#ifndef UNPACK_PHYSICAL_ROWS
#define UNPACK_PHYSICAL_ROWS UNPACK_LOGICAL_ROWS
#endif
#ifndef UNPACK_LOGICAL_COLUMNS
#define UNPACK_LOGICAL_COLUMNS 16
#endif
#ifndef UNPACK_PHYSICAL_COLUMNS
#define UNPACK_PHYSICAL_COLUMNS UNPACK_LOGICAL_COLUMNS
#endif
#ifndef UNPACK_VERTEX_NAME
#define UNPACK_VERTEX_NAME UnpackAmpToRowMajorF16
#endif

static_assert(UNPACK_SOURCE_ORDER == 0 || UNPACK_SOURCE_ORDER == 1);
static_assert(UNPACK_SOURCE_ORDER == 0 || UNPACK_PHYSICAL_ROWS % 16 == 0);
static_assert(UNPACK_PHYSICAL_COLUMNS % 16 == 0 || UNPACK_SOURCE_ORDER == 1);
static_assert(UNPACK_PHYSICAL_COLUMNS % 2 == 0);

class UNPACK_VERTEX_NAME : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned matrices;
  unsigned logicalRows;
  unsigned physicalRows;
  unsigned logicalColumns;
  unsigned physicalColumns;

  bool compute(unsigned worker) {
    constexpr unsigned logicalRowCount = UNPACK_LOGICAL_ROWS;
    constexpr unsigned physicalRowCount = UNPACK_PHYSICAL_ROWS;
    constexpr unsigned logicalColumnCount = UNPACK_LOGICAL_COLUMNS;
    constexpr unsigned physicalColumnCount = UNPACK_PHYSICAL_COLUMNS;
    const unsigned *sourceWords = reinterpret_cast<const unsigned *>(&source[0]);
    unsigned *destinationWords = reinterpret_cast<unsigned *>(&destination[0]);
    const unsigned matrixElements = physicalRowCount * physicalColumnCount;
    for (unsigned matrixRow = worker;
         matrixRow < matrices * physicalRowCount; matrixRow += 6) {
      const unsigned matrix = matrixRow / physicalRowCount;
      const unsigned row = matrixRow % physicalRowCount;
      const unsigned matrixBase = matrix * matrixElements;
      for (unsigned column = 0; column < physicalColumnCount; column += 2) {
        unsigned packed = 0;
        if (row < logicalRowCount && column < logicalColumnCount) {
          for (unsigned lane = 0; lane < 2; ++lane) {
            const unsigned semanticColumn = column + lane;
            if (semanticColumn >= logicalColumnCount)
              continue;
            unsigned physical;
#if UNPACK_SOURCE_ORDER == 0
            const unsigned logicalPair = semanticColumn % 16 / 2;
            const unsigned physicalPair = logicalPair % 4 * 2 + logicalPair / 4;
            const unsigned physicalColumn = physicalPair * 2 + semanticColumn % 2;
            physical = (semanticColumn / 16) * physicalRowCount * 16 + row * 16 +
                       physicalColumn;
#else
            physical = (row / 16) * physicalColumnCount * 16 +
                       semanticColumn * 16 + row % 16;
#endif
            const unsigned sourceWord = sourceWords[(matrixBase + physical) / 2];
            const unsigned value =
                (sourceWord >> (((matrixBase + physical) & 1) * 16)) & 0xffff;
            packed |= value << (lane * 16);
          }
        }
        destinationWords[(matrixBase + row * physicalColumnCount + column) / 2] =
            packed;
      }
    }
    return true;
  }
};
