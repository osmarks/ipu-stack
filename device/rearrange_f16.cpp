#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

#ifndef REARRANGE_LOGICAL_COLUMNS
#define REARRANGE_LOGICAL_COLUMNS 64
#endif
#ifndef REARRANGE_PHYSICAL_COLUMNS
#define REARRANGE_PHYSICAL_COLUMNS REARRANGE_LOGICAL_COLUMNS
#endif
#ifndef REARRANGE_VERTEX_NAME
#define REARRANGE_VERTEX_NAME RearrangeRowMajorToAmpF16
#endif

static_assert(REARRANGE_LOGICAL_COLUMNS % 2 == 0);

class REARRANGE_VERTEX_NAME : public MultiVertex {
public:
  Input<Vector<half, VectorLayout::ONE_PTR>> source;
  Output<Vector<half, VectorLayout::ONE_PTR>> destination;
  unsigned logicalRows;
  unsigned physicalRows;
  unsigned targetOrder;

  bool compute(unsigned worker) {
    constexpr unsigned inner = 8;
    constexpr unsigned innerBlock = 64;
    constexpr unsigned columnBlock = 16;
    const unsigned *sourceWords = reinterpret_cast<const unsigned *>(&source[0]);
    unsigned *destinationWords = reinterpret_cast<unsigned *>(&destination[0]);
    if (targetOrder < 2) {
      for (unsigned row = worker; row < physicalRows; row += 6) {
        const unsigned logicalPair = row % columnBlock / 2;
        const unsigned loadPair = logicalPair % 4 * 2 + logicalPair / 4;
        const unsigned loadChannel = loadPair * 2 + row % 2;
        for (unsigned column = 0; column < REARRANGE_PHYSICAL_COLUMNS;
             column += 2) {
          unsigned low = 0;
          unsigned high = 0;
          if (row < logicalRows && column < REARRANGE_LOGICAL_COLUMNS) {
            const unsigned sourceElement =
                row * REARRANGE_LOGICAL_COLUMNS + column;
            low = sourceWords[sourceElement / 2] & 0xffff;
            high = sourceWords[sourceElement / 2] >> 16;
          }
          unsigned physical;
          if (targetOrder == 0) {
            physical = (column / inner) * physicalRows * inner + row * inner +
                       column % inner;
          } else {
            const unsigned panel =
                (row / columnBlock) * (REARRANGE_PHYSICAL_COLUMNS / inner) +
                column / inner;
            physical = panel * inner * columnBlock + loadChannel * inner +
                       column % inner;
          }
          destinationWords[physical / 2] = low | (high << 16);
        }
      }
      return true;
    }
    const unsigned rowPairs = (physicalRows + 1) / 2;
    for (unsigned rowPair = worker; rowPair < rowPairs; rowPair += 6) {
      const unsigned row = rowPair * 2;
      for (unsigned column = 0; column < REARRANGE_PHYSICAL_COLUMNS; ++column) {
        const unsigned logicalPair = column % columnBlock / 2;
        const unsigned loadPair = logicalPair % 4 * 2 + logicalPair / 4;
        const unsigned loadChannel = loadPair * 2 + column % 2;
        const unsigned innerGroup = row % innerBlock / inner;
        const unsigned panel =
          (row / innerBlock) * (REARRANGE_PHYSICAL_COLUMNS / columnBlock) *
              (innerBlock / inner) +
          (column / columnBlock) * (innerBlock / inner) + innerGroup;
        const unsigned physical =
            panel * inner * columnBlock + loadChannel * inner + row % inner;
        unsigned low = 0;
        unsigned high = 0;
        if (row < logicalRows && column < REARRANGE_LOGICAL_COLUMNS) {
          const unsigned sourceElement = row * REARRANGE_LOGICAL_COLUMNS + column;
          const unsigned shift = (sourceElement % 2) * 16;
          low = (sourceWords[sourceElement / 2] >> shift) & 0xffff;
          if (row + 1 < logicalRows) {
            const unsigned next = sourceElement + REARRANGE_LOGICAL_COLUMNS;
            high = (sourceWords[next / 2] >> shift) & 0xffff;
          }
        }
        destinationWords[physical / 2] = low | (high << 16);
      }
    }
    return true;
  }
};
