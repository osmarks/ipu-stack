#include <cstdint>
#include <iomanip>
#include <iostream>

#include <ipu_arch_info/ipuArchInfo.h>

namespace {

void print(const IPUArchInfo &arch, const char *name, std::uint32_t word) {
  std::cout << std::left << std::setw(24) << name << " 0x" << std::right
            << std::hex << std::setw(8) << std::setfill('0') << word
            << std::setfill(' ') << "  "
            << arch.disassembler.disassemble(true, word) << '\n';
}

} // namespace

int main() {
  const auto &arch = ipuArchInfoByName("ipu21");
  for (const auto address : {0u, 1u, 0x14021u}) {
    const auto name = "sendpicp address=" + std::to_string(address);
    print(arch, name.c_str(),
          arch.encode.sendpicp_mmmn_zi_bf_zi_zi(63, address, 0, 0));
  }
  for (unsigned sctl = 0; sctl != 8; ++sctl) {
    const auto name = "sendpicp sctl=" + std::to_string(sctl);
    print(arch, name.c_str(),
          arch.encode.sendpicp_mmmn_zi_bf_zi_zi(63, 0, sctl, 0));
  }
  for (unsigned picSelector = 0; picSelector != 2; ++picSelector) {
    const auto name = "sendpicp pic-selector=" + std::to_string(picSelector);
    print(arch, name.c_str(),
          arch.encode.sendpicp_mmmn_zi_bf_zi_zi(0, 0, 0, picSelector));
  }
  for (unsigned sctl = 0; sctl != 8; ++sctl) {
    const auto name = "send sctl=" + std::to_string(sctl);
    print(arch, name.c_str(),
          arch.encode.send_mmmn_zi_bf_zi(63, 0x14000, sctl));
  }
  for (const auto word : {0x7b6a0241u, 0x70f00640u, 0x71f95000u,
                          0xf7e00018u, 0x03018000u, 0xf54a0109u,
                          0x19015000u}) {
    print(arch, "oracle word", word);
  }
}
