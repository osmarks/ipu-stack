#!/usr/bin/env python3
import sys

from IPUArchInfo_py3 import ipuArchInfoByName


arch = ipuArchInfoByName("ipu21")
for argument in sys.argv[1:]:
    word = int(argument, 0)
    print(f"{word:08x}  {arch.disassembler.disassemble(True, word)}")
