#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

int main(int argc, char **argv) try {
  if (argc != 4)
    return 2;
  constexpr std::size_t transfers = 4;
  const std::size_t words = std::stoul(argv[2]);
  const unsigned destinationTile = std::stoul(argv[3]);
  poplar::DeviceManager manager;
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  if (devices.empty() || !devices.front().attach())
    return 3;
  poplar::Graph graph(devices.front().getTarget());
  graph.addCodelets("codelets.cpp", poplar::CodeletFileType::Auto, "-O2",
                    "ipu2");
  auto source =
      graph.addVariable(poplar::UNSIGNED_INT, {transfers, words}, "source");
  std::vector<poplar::Tensor> destination;
  for (std::size_t i = 0; i < transfers; ++i) {
    graph.setTileMapping(source[i], 5 + i);
    destination.push_back(
        graph.addVariable(poplar::UNSIGNED_INT, {words}, "destination"));
    graph.setTileMapping(destination.back(), destinationTile);
  }
  graph.createHostWrite("source-write", source.flatten());
  for (std::size_t i = 0; i < transfers; ++i)
    graph.createHostRead("destination-read-" + std::to_string(i),
                         destination[i]);
  auto keep = graph.addComputeSet("keep-separated");
  auto vertex = graph.addVertex(keep, "KeepSeparated");
  for (std::size_t i = 0; i < transfers; ++i)
    graph.connect(vertex["d" + std::to_string(i)], destination[i]);
  graph.setTileMapping(vertex, destinationTile);
  poplar::program::Sequence program;
  for (std::size_t i = 0; i < transfers; ++i)
    program.add(poplar::program::Copy(source[i], destination[i]));
  program.add(poplar::program::Execute(keep));
  poplar::Engine engine(graph, program);
  std::ofstream executable(argv[1], std::ios::binary);
  engine.serializeExecutable(executable);
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
