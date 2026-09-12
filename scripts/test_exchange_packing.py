"""Semantic reconstruction tests for the offline transport experiment."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "packing", Path(__file__).with_name("exchange-packing.py")
)
packing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing)


def transfer(source, addresses, destinations, words=2):
    return {
        "source": source,
        "source_addresses": addresses,
        "words": words,
        "width": "Word32",
        "destinations": [{"tile": t, "address": a} for t, a in destinations],
    }


class PackingTests(unittest.TestCase):
    def test_source_destination_and_joint_fragmentation(self):
        for source_step, destination_step, expected in [
            (16, 8, [1, 3, 1]),
            (8, 16, [3, 1, 1]),
            (16, 16, [3, 3, 1]),
        ]:
            phase = {
                "phase": 0,
                "transfers": [
                    transfer(
                        0,
                        [0x50000 + i * source_step],
                        [
                            (1, 0x60000 + i * destination_step),
                            (2, 0x70000 + i * destination_step),
                        ],
                    )
                    for i in range(3)
                ],
            }
            for mode, count in zip(("source", "destination", "both"), expected):
                result, metrics = packing.transform(phase, mode)
                self.assertEqual(len(result["transfers"]), count)
                self.assertEqual(packing.geometry(result)["received_bytes"], 48)
                self.assertEqual(packing.geometry(result)["transmitted_bytes"], 24)
                self.assertGreater(metrics["copy_cycles_floor"], 0)
            original, _ = packing.transform(phase, "original")
            self.assertIs(original, phase)

    def test_reconstructs_received_payloads(self):
        phase = {
            "phase": 7,
            "transfers": [
                transfer(
                    0,
                    [0x50000 + i * 24],
                    [(1, 0x60000 + i * 16), (2, 0x70000 + i * 16)],
                )
                for i in range(9)
            ]
            + [transfer(2, [0x51000], [(1, 0x61000)], 3)],
        }
        initial = {}
        expected = {}
        for t in phase["transfers"]:
            for word in range(t["words"]):
                address = t["source_addresses"][0] + 4 * word
                token = (t["source"], address)
                initial[token] = token
                for d in t["destinations"]:
                    expected[d["tile"], d["address"] + 4 * word] = token
        for mode in ("source", "destination", "both"):
            result, info = packing.transform(phase, mode, include_copies=True)
            memory = initial.copy()

            def copy(side, info=info, memory=memory):
                for tile, copies in info["copies"][side].items():
                    for source, destination, size in copies:
                        for offset in range(0, size, 4):
                            memory[tile, destination + offset] = memory[
                                tile, source + offset
                            ]

            copy("source")
            for t in result["transfers"]:
                for word in range(t["words"]):
                    token = memory[t["source"], t["source_addresses"][0] + 4 * word]
                    for d in t["destinations"]:
                        memory[d["tile"], d["address"] + 4 * word] = token
            copy("destination")
            for address, token in expected.items():
                self.assertEqual(memory[address], token)

    def test_repeat_contiguity_must_hold_in_every_iteration(self):
        phase = {
            "phase": 1,
            "transfers": [
                transfer(0, [0x50000, 0x60000], [(1, 0x70000)]),
                transfer(0, [0x50008, 0x60010], [(1, 0x70008)]),
            ],
        }
        destination, _ = packing.transform(phase, "destination")
        self.assertEqual(len(destination["transfers"]), 2)
        both, _ = packing.transform(phase, "both")
        self.assertEqual(len(both["transfers"]), 1)

    def test_large_runs_respect_snapshot_word_limit(self):
        phase = {
            "phase": 0,
            "transfers": [
                transfer(
                    0, [0x50000 + i * 8192], [(1, 0x60000 + i * 16384)], words=2048
                )
                for i in range(4)
            ],
        }
        result, _ = packing.transform(phase, "destination")
        self.assertEqual([t["words"] for t in result["transfers"]], [4148, 4044])
        self.assertEqual(
            result["transfers"][1]["source_addresses"], [0x50000 + 4148 * 4]
        )

    def test_different_recipient_sets_stay_separate(self):
        phase = {
            "phase": 0,
            "transfers": [
                transfer(0, [0x50000], [(1, 0x60000)]),
                transfer(0, [0x50008], [(2, 0x60000)]),
            ],
        }
        result, _ = packing.transform(phase, "both")
        self.assertEqual(len(result["transfers"]), 2)

    def test_contiguous_copies_and_affine_loops(self):
        copies = []
        packing.append_copy(copies, 0, 100, 4)
        packing.append_copy(copies, 4, 104, 8)
        self.assertEqual(copies, [(0, 100, 12)])
        self.assertEqual(packing.copy_loops([(0, 0, 8), (16, 8, 8), (32, 16, 8)]), 1)


if __name__ == "__main__":
    unittest.main()
