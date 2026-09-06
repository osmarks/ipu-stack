#!/usr/bin/env python3
"""Replay multicast boundary regressions on an IPU (one device owner at a time)."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sdk', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--config', default='c600-init.ipucfg')
parser.add_argument('--binary', default='target/release/ipu-trivial-test')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)


def transfer(source, receivers, words, index, width='Word32'):
    return dict(source=source, source_addresses=[0x80000 + index * 0x1000],
                destinations=[dict(tile=tile, address=0x90000 + index * 0x1000)
                              for tile in receivers], words=words, width=width)


def replay(name, transfers):
    snapshot = args.output / (name + '.json')
    snapshot.write_text(json.dumps(dict(schema_version=3, tile_count=1472,
                                       phases=[dict(phase=0, transfers=transfers)])))
    with (args.output / (name + '.log')).open('w') as log:
        result = subprocess.run([args.binary, args.config, '--sdk', args.sdk,
                                 '--package', str(args.output / (name + '.ipuexe')),
                                 '--replay-exchange-schedule', str(snapshot),
                                 '--exchange-replay-phase', '0',
                                 '--exchange-replay-samples', '4096'],
                                stdout=log, stderr=subprocess.STDOUT, timeout=90)
    print(name, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
    return result.returncode == 0


passed = True
for words in [1, 16, 51, 52, 53, 64, 65, 352, 368]:
    passed &= replay(f'word-{words}', [
        transfer(source, range(10, 64), words, i)
        for i, source in enumerate([0, 4, 6, 8])])
for words in [128, 352, 368]:
    passed &= replay(f'paired-{words}', [
        transfer(source, range(10, 64), words, i, 'Paired64')
        for i, source in enumerate([0, 4, 6, 8])])
for width in ['Word32', 'Paired64']:
    passed &= replay(f'far-{width}', [
        transfer(source, range(10, 64), 352, i, width)
        for i, source in enumerate([0, 736, 1286, 1470])])
for width in ['Word32', 'Paired64']:
    passed &= replay(f'same-source-{width}', [
        transfer(0, range(10, 64), 352, i, width) for i in range(4)])
for receiver in [2, 3, 46, 47]:
    passed &= replay(f'mixed-{receiver}', [
        transfer(0, [receiver], 972, 0),
        transfer(4, [receiver & ~1, receiver | 1], 352, 1, 'Paired64')])
raise SystemExit(0 if passed else 1)
