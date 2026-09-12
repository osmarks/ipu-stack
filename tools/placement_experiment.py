#!/usr/bin/env python3
"""Offline placement experiments. No compiler or device state is changed.

Accepts placement-constraint JSON dumps; randomized cases include a checked
placement witness, so a heuristic failure is never called infeasibility.
"""
import argparse
import heapq
import json
import random
import statistics
import time
from pathlib import Path

MAX = 2**32 - 1
R1 = 524288
HOST = (327680, 360448)


def align(n, a):
    return (n + a - 1) // a * a


def overlaps(a, b, c, d):
    return a < d and c < b


def live(r, q):
    return r['lifetime']['first'] <= q['lifetime']['last'] and q['lifetime']['first'] <= r['lifetime']['last']


def elements(data, a, b):
    boundary = data.get('region_boundary', R1)
    return [(lo // size * size, align(hi, size))
            for lo, hi, size in [(a, min(b, boundary), data.get('region0_element_bytes', 16384)),
                                 (max(a, boundary), b, data.get('region1_element_bytes', 32768))]
            if lo < hi]


def subtract(ranges, a, b):
    result = []
    for lo, hi in ranges:
        if not overlaps(lo, hi, a, b):
            result.append((lo, hi))
        else:
            if lo < a:
                result.append((lo, a))
            if b < hi:
                result.append((b, hi))
    return result


def domains(data, r):
    first, last = r['lifetime']['first'], r['lifetime']['last']
    boundary = data.get('region_boundary', R1)
    host = data.get('host_scratch_range', HOST)
    for lo, hi in data['ranges']:
        if host[0] <= lo and hi <= host[1] and (first == 0 or last == MAX):
            continue
        for region in (0, 1):
            if not region and r['class'] == 'Ipu21Interleaved':
                continue
            start = max(lo, boundary + (data['interleaved_offset'] if r['class'] == 'Ipu21Interleaved' else 0)) if region else lo
            end = hi if region else min(hi, boundary)
            alignment = max(r['alignment'], data.get('region1_element_bytes' if region else 'region0_element_bytes', 32768 if region else 16384) if r['region1_stride'] is not None else 1)
            size = r['region1_stride'] * len(r['assignments']) if region and r['region1_stride'] is not None else r['bytes']
            if align(start, alignment) + size <= end:
                yield region, start, end, alignment, size


def conflict_sets(data):
    owner = {root: i for i, r in enumerate(data['requests']) for root, _ in r['assignments']}
    edges = [set() for _ in data['requests']]
    for i, r in enumerate(data['requests']):
        for root in r['conflicts']:
            if root in owner and owner[root] != i:
                edges[i].add(owner[root])
                edges[owner[root]].add(i)
    return edges


def validate(data, placement):
    """Independent checks: containment, size, alignment, lifetimes and banks."""
    requests = data['requests']
    assert len(placement) == len(requests)
    edges = conflict_sets(data)
    for i, (a, b) in placement.items():
        r = requests[i]
        assert any(lo <= a and b <= hi and b - a == size and a % alignment == 0
                   for _, lo, hi, alignment, size in domains(data, r)), (i, a, b, 'domain')
        for j, (c, d) in placement.items():
            if j >= i:
                continue
            assert not live(r, requests[j]) or not overlaps(a, b, c, d), (i, j, 'lifetime')
            if j in edges[i]:
                assert all(not overlaps(a, b, lo, hi) for lo, hi in elements(data, c, d)), (i, j, 'bank')


def peak(data):
    requests = data['requests']
    # Region-dependent stride can only increase this bound for real fixtures.
    sizes = [min((d[4] for d in domains(data, r)), default=r['bytes']) for r in requests]
    return max((sum(size for r, size in zip(requests, sizes)
                    if r['lifetime']['first'] <= t <= r['lifetime']['last'])
                for t in {r['lifetime']['first'] for r in requests}), default=0)


def order_for(data, mode):
    def key(i):
        r = data['requests'][i]
        first, last = r['lifetime']['first'], r['lifetime']['last']
        a, b = r['alignment'], r['bytes']
        if mode == 'time':
            return first, r['class'] != 'Ipu21Interleaved', -a, -b, last, i
        if mode == 'alignment':
            return -a, -b, -(last - first), first, i
        if mode == 'end':
            return last != MAX, -b, -a, first, i
        raise ValueError(mode)
    return tuple(sorted(range(len(data['requests'])), key=key))


def greedy(data, order, edges, fit='first'):
    placed = {}
    for i in order:
        r = data['requests'][i]
        candidates = []
        for region, lo, hi, alignment, size in domains(data, r):
            free = [(lo, hi)]
            for j, (a, b) in placed.items():
                if live(r, data['requests'][j]):
                    free = subtract(free, a, b)
                if j in edges[i]:
                    for c, d in elements(data, a, b):
                        free = subtract(free, c, d)
            for a, b in free:
                start = align(a, alignment)
                if start + size <= b:
                    score = (region, start) if fit == 'first' else (region, b - start - size, start)
                    candidates.append((score, start, start + size))
        if not candidates:
            blockers = [j for j in placed if live(r, data['requests'][j]) or j in edges[i]]
            return placed, i, blockers
        _, a, b = min(candidates)
        placed[i] = (a, b)
    validate(data, placed)
    return placed, None, []


def solve(data, budget=64):
    """Bounded failure-directed ordering search, not an exact solver.

Try current orders and an end-of-model-first order. On failure, move the
failed request before a few actual blockers and retry. Prefer branches that
placed more requests, deduplicate orders, and never alter constraints.
"""
    start = time.perf_counter()
    capacity = sum(b - a for a, b in data['ranges'])
    lower = peak(data)
    if lower > capacity:
        return {'status': 'infeasible_bound', 'deficit': lower - capacity, 'trials': 0,
                'seconds': time.perf_counter() - start}
    edges = conflict_sets(data)
    queue, visited = [], set()
    serial = 0
    for mode in ['time', 'alignment', 'end']:
        heapq.heappush(queue, (-len(data['requests']) - 1, serial, order_for(data, mode)))
        serial += 1
    trials = 0
    while queue and trials < budget:
        _, _, order = heapq.heappop(queue)
        if order in visited:
            continue
        visited.add(order)
        placed, failed, blockers = greedy(data, order, edges)
        trials += 1
        if failed is None:
            return {'status': 'fit', 'trials': trials, 'seconds': time.perf_counter() - start}
        positions = {i: p for p, i in enumerate(order)}
        blockers.sort(key=lambda i: positions[i])
        # Earliest blocker, middle blocker, latest blocker, and largest blocker.
        choices = blockers[:1] + blockers[len(blockers)//2:len(blockers)//2+1] + blockers[-1:]
        if blockers:
            choices.append(max(blockers, key=lambda i: data['requests'][i]['bytes']))
        for blocker in set(choices):
            revised = list(order)
            revised.remove(failed)
            revised.insert(positions[blocker], failed)
            heapq.heappush(queue, (-len(placed), serial, tuple(revised)))
            serial += 1
    return {'status': 'unknown', 'trials': trials, 'seconds': time.perf_counter() - start}


def synthetic(seed):
    """Known-feasible dense interval instances; no heuristic selects the witness.

Each physical element is divided into several spatial lanes. Every lane has
random sequential lifetimes and aligned, nearly full-width buffers. Bank
edges are sampled only across distinct witness elements. Requests are then
shuffled; the solver never receives witness positions. A ninth element supplies 12.5%
    capacity headroom; eight-element variants test the tight limit.
"""
    rng = random.Random(seed)
    data = {'tile': 0, 'ranges': [[R1, R1 + 9*32768]], 'interleaved_offset': 0, 'requests': []}
    witness = {}
    for bank in range(8):
        lanes = rng.choice([1, 2, 4])
        width = 32768 // lanes
        for lane in range(lanes):
            t = 1
            while t < 48:
                end = min(48, t + rng.randint(4, 14))
                index = len(data['requests'])
                size = align(rng.randint(width * 3 // 4, width), 32)
                r = {'class': rng.choice(['Ipu21Standard', 'Ipu21Interleaved']), 'region1_stride': None,
                     'lifetime': {'first': t, 'last': end, 'seen': True}, 'bytes': size,
                     'alignment': rng.choice([8, 32, 64]), 'assignments': [[index, 0]], 'conflicts': []}
                data['requests'].append(r)
                witness[index] = (R1 + bank*32768 + lane*width, R1 + bank*32768 + lane*width + size)
                t = end + 1
    requests = data['requests']
    for i, r in enumerate(requests):
        for j in range(i):
            if witness[i][0] // 32768 != witness[j][0] // 32768 and live(r, requests[j]) and rng.random() < .012:
                r['conflicts'].append(j)
                requests[j]['conflicts'].append(i)
    permutation = list(range(len(requests)))
    rng.shuffle(permutation)
    data['requests'] = [requests[i] for i in permutation]
    witness = {new: witness[old] for new, old in enumerate(permutation)}
    validate(data, witness)
    return data


def evaluate(data, budget):
    edges = conflict_sets(data)
    results = {}
    for mode in ['time', 'alignment', 'end']:
        start = time.perf_counter()
        placed, failed, _ = greedy(data, order_for(data, mode), edges)
        results[mode] = {'fit': failed is None, 'placed': len(placed), 'seconds': time.perf_counter() - start}
    results['repair'] = solve(data, budget)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='*', type=Path)
    parser.add_argument('--random-cases', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--budget', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for path in args.captures:
        data = json.loads(path.read_text())
        if data.get('placed'):
            roots = data['assigned_root_spans']
            validate(data, {i: tuple(roots[str(r['assignments'][0][0])])
                            for i, r in enumerate(data['requests'])})
        results[str(path)] = evaluate(data, args.budget)
    for seed in range(args.seed, args.seed + args.random_cases):
        data = synthetic(seed)
        results[f'synthetic-{seed}'] = evaluate(data, args.budget)
        if seed < args.seed + 10:
            data['ranges'][0][1] -= 32768
            results[f'tight-{seed}'] = evaluate(data, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'seed': args.seed, 'budget': args.budget, 'results': results}, indent=2) + '\n')
    for label, group in [('captured', [v for k, v in results.items() if not k.startswith(('synthetic', 'tight'))]),
                         ('synthetic', [v for k, v in results.items() if k.startswith('synthetic')]),
                         ('tight', [v for k, v in results.items() if k.startswith('tight')])]:
        print(label, len(group), 'current combined fits',
              sum(v['time']['fit'] or v['alignment']['fit'] for v in group))
        for mode in ['time', 'alignment', 'end', 'repair']:
            print(mode, 'fits', sum(v[mode].get('fit', v[mode].get('status') == 'fit') for v in group),
                  'seconds', round(sum(v[mode]['seconds'] for v in group), 3))
        times = sorted(v['repair']['seconds'] for v in group)
        if times:
            print('repair median/p95/max seconds', statistics.median(times),
                  times[int(.95 * (len(times) - 1))], max(times))


if __name__ == '__main__':
    main()
