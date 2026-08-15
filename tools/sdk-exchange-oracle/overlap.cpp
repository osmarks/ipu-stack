#include <cstdint>
#include <fstream>
#include <iostream>

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>

int main(int argc, char **argv) try {
  if (argc != 7)
    return 2;
  const std::size_t words = std::stoul(argv[2]);
  const unsigned incomingSourceTile = std::stoul(argv[3]);
  const unsigned pivotTile = std::stoul(argv[4]);
  const unsigned outgoingDestinationTile = std::stoul(argv[5]);
  const bool reverseCopies = std::stoul(argv[6]) != 0;

  poplar::DeviceManager manager;
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  if (devices.empty() || !devices.front().attach())
    return 3;
  poplar::Graph graph(devices.front().getTarget());
  graph.addCodelets("codelets.cpp", poplar::CodeletFileType::Auto, "-O2",
                    "ipu2");

  auto incomingSource =
      graph.addVariable(poplar::UNSIGNED_INT, {words}, "incoming-source");
  auto pivotReceive =
      graph.addVariable(poplar::UNSIGNED_INT, {words}, "pivot-receive");
  auto pivotSend =
      graph.addVariable(poplar::UNSIGNED_INT, {words}, "pivot-send");
  auto outgoingDestination = graph.addVariable(
      poplar::UNSIGNED_INT, {words}, "outgoing-destination");
  graph.setTileMapping(incomingSource, incomingSourceTile);
  graph.setTileMapping(pivotReceive, pivotTile);
  graph.setTileMapping(pivotSend, pivotTile);
  graph.setTileMapping(outgoingDestination, outgoingDestinationTile);

  auto keep = graph.addComputeSet("keep-pivot-buffers-separated");
  auto vertex = graph.addVertex(keep, "KeepPairSeparated");
  graph.connect(vertex["incoming"], pivotReceive);
  graph.connect(vertex["outgoing"], pivotSend);
  graph.setTileMapping(vertex, pivotTile);

  graph.createHostWrite("incoming-source-write", incomingSource);
  graph.createHostWrite("pivot-send-write", pivotSend);
  graph.createHostRead("pivot-receive-read", pivotReceive);
  graph.createHostRead("outgoing-destination-read", outgoingDestination);

  poplar::program::Sequence program;
  if (reverseCopies) {
    program.add(poplar::program::Copy(pivotSend, outgoingDestination));
    program.add(poplar::program::Copy(incomingSource, pivotReceive));
  } else {
    program.add(poplar::program::Copy(incomingSource, pivotReceive));
    program.add(poplar::program::Copy(pivotSend, outgoingDestination));
  }
  program.add(poplar::program::Execute(keep));
  poplar::Engine engine(graph, program);
  std::ofstream executable(argv[1], std::ios::binary);
  engine.serializeExecutable(executable);
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
