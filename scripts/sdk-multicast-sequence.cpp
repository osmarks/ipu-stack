#include <cstdint>
#include <fstream>
#include <iostream>
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <vector>
int main(int argc, char **argv) try {
  // Usage: executable OUTPUT WORDS [SECOND_RECEIVER]; 3 enables paired RX.
  if (argc < 3 || argc > 4)
    return 2;
  const unsigned words = std::stoul(argv[2]);
  const unsigned receiver = argc == 4 ? std::stoul(argv[3]) : 5;
  auto manager = poplar::DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
  if (devices.empty() || !devices[0].attach())
    return 3;
  poplar::Graph graph(devices[0].getTarget());
  auto source = graph.addVariable(poplar::UNSIGNED_INT, {4, words}, "source");
  const unsigned senders[] = {0, 4, 6, 8};
  poplar::program::Sequence program;
  graph.createHostWrite("source", source.flatten());
  for (unsigned i = 0; i < 4; ++i) {
    graph.setTileMapping(source[i], senders[i]);
    auto out = graph.addVariable(poplar::UNSIGNED_INT, {2, words},
                                 "destination" + std::to_string(i));
    graph.setTileMapping(out[0], 2);
    graph.setTileMapping(out[1], receiver);
    graph.createHostRead("output" + std::to_string(i), out.flatten());
    program.add(
        poplar::program::Copy(source[i].expand({0}).broadcast(2, 0), out));
  }
  poplar::OptionFlags opts;
  opts.set("exchange.multicastPolicy", "balanced");
  poplar::Engine engine(graph, program, opts);
  std::ofstream file(argv[1], std::ios::binary);
  engine.serializeExecutable(file);
  file.close();
  std::vector<unsigned> data(4 * words), result(2 * words);
  for (unsigned i = 0; i < data.size(); ++i)
    data[i] = 0x51a70000u ^ (i * 0x9e3779b9u);
  engine.load(devices[0]);
  engine.writeTensor("source", data.data(), data.data() + data.size());
  engine.run();
  for (unsigned i = 0; i < 4; ++i) {
    engine.readTensor("output" + std::to_string(i), result.data(),
                      result.data() + result.size());
    for (unsigned j = 0; j < result.size(); ++j)
      if (result[j] != data[i * words + j % words])
        throw std::runtime_error("data mismatch");
  }
  std::cout << "SDK multicast sequence PASS " << 8 * words << " words\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
