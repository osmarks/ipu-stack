#include <poplar/Vertex.hpp>

class [[poplar::constraint("elem(*d0) != elem(*d1)"),
        poplar::constraint("elem(*d0) != elem(*d2)"),
        poplar::constraint("elem(*d0) != elem(*d3)"),
        poplar::constraint("elem(*d1) != elem(*d2)"),
        poplar::constraint("elem(*d1) != elem(*d3)"),
        poplar::constraint("elem(*d2) != elem(*d3)")]]
KeepSeparated : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<unsigned>> d0;
  poplar::InOut<poplar::Vector<unsigned>> d1;
  poplar::InOut<poplar::Vector<unsigned>> d2;
  poplar::InOut<poplar::Vector<unsigned>> d3;
  bool compute() { return true; }
};

class [[poplar::constraint("elem(*incoming) != elem(*outgoing)")]]
KeepPairSeparated : public poplar::Vertex {
public:
  poplar::InOut<poplar::Vector<unsigned>> incoming;
  poplar::InOut<poplar::Vector<unsigned>> outgoing;
  bool compute() { return true; }
};
