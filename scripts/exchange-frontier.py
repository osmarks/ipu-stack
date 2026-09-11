#!/usr/bin/env python3
"""Compare scheduler policies on fixed captured transfers; render Pareto curves."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess


def frontier(rows, storage):
    return [r for r in rows if not any(
        q[storage] <= r[storage] and q['cycles'] <= r['cycles']
        and (q[storage] < r[storage] or q['cycles'] < r['cycles']) for q in rows)]


def render(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'svg.fonttype': 'none'})
    phases = sorted({r['phase'] for r in rows})
    colors = {'priority': '#555555', 'stream': '#176ba0', 'balanced': '#c34f18'}
    for storage, label in [('maximum_row_bytes', 'Maximum row bytes per tile'),
                           ('total_row_bytes', 'Total row bytes across tiles')]:
        fig, axes = plt.subplots((len(phases)+1)//2, 2, figsize=(13, 3.5*((len(phases)+1)//2)), squeeze=False)
        for ax, phase in zip(axes.flat, phases):
            points = [r for r in rows if r['phase'] == phase]
            best = frontier(points, storage)
            for family, color in colors.items():
                group = [r for r in points if r['family'] == family]
                ax.scatter([r[storage] for r in group], [r['cycles'] for r in group], s=25,
                           color=color, label=family, alpha=.8)
            unique = sorted({(r[storage], r['cycles']) for r in best})
            ax.plot([x for x, y in unique], [y for x, y in unique], color='black', linewidth=.9)
            for x, y in unique:
                matches = [r['configuration'] for r in best if (r[storage], r['cycles']) == (x, y)]
                names = '/'.join(matches[:2]) + (f' (+{len(matches)-2})' if len(matches)>2 else '')
                ax.annotate(names, (x, y), xytext=(4, 5), textcoords='offset points', fontsize=6)
            ax.set_xscale('log', base=2)
            ax.set_xlabel(label)
            ax.set_ylabel('Scheduled cycles after barrier')
            ax.set_title(f"Phase {phase}: {points[0]['transfers']:,} transfers")
            ax.grid(True, linewidth=.3, color='#bbbbbb')
        for ax in list(axes.flat)[len(phases):]:
            ax.remove()
        axes.flat[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output / f'{storage}.svg')
        fig.savefig(output / f'{storage}.pdf')
        plt.close(fig)
    chosen = []
    for phase in phases:
        points = [r for r in rows if r['phase'] == phase]
        for r in points:
            r['maximum_row_frontier'] = r in frontier(points, 'maximum_row_bytes')
            r['total_row_frontier'] = r in frontier(points, 'total_row_bytes')
            if r['maximum_row_frontier'] or r['total_row_frontier']:
                chosen.append(r)
    for name, data in [('results', rows), ('frontier', chosen)]:
        (output / f'{name}.json').write_text(json.dumps(data, indent=2)+'\n')
        with (output / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / 'index.html').write_text('''<!doctype html><meta charset="utf-8">
<title>Exchange scheduling frontiers</title>
<style>body{max-width:1400px;margin:2em auto;padding:0 1em;font-family:Georgia,serif;color:#111;background:white}img{width:100%}a{color:inherit}</style>
<h1>Exchange scheduling frontiers</h1>
<p>Fixed captured transfers and addresses. Scheduled cycles exclude waiting for the barrier.
The black line joins nondominated points. <a href="results.csv">All measurements</a> ·
<a href="frontier.csv">Frontier points</a> · <a href="manifest.json">Commands</a></p>
<img src="maximum_row_bytes.svg" alt="Cycles versus maximum row storage per tile">
<img src="total_row_bytes.svg" alt="Cycles versus total row storage">
''')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/ipu-exchange-schedule-bench'))
    parser.add_argument('--words', default='64,128,256,512,1024,4096,16384')
    parser.add_argument('--phase', type=int, action='append', default=[])
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        render(json.loads((args.output/'results.json').read_text()), args.output)
        return
    configs = [(p, 'priority', ['--priority', p]) for p in
               ['automatic', 'combined', 'directional', 'remaining-combined', 'remaining-directional']]
    for words in map(int, args.words.split(',')):
        if words <= 0:
            parser.error('chunk sizes must be positive')
        configs.extend([(f'stream-{words}', 'stream', ['--stream-words', str(words)]),
                        (f'balanced-{words}', 'balanced', ['--stream-words', str(words), '--balance-streams'])])
    commands = []
    rows = []
    for name, family, flags in configs:
        command = [str(args.binary.resolve()), str(args.snapshot.resolve()), *flags]
        for phase in args.phase:
            command.extend(['--phase', str(phase)])
        commands.append(command)
        log = args.output/f'{name}.log'
        print(name, flush=True)
        # Sequential runs and a single Rayon worker keep compiler-time comparisons
        # interpretable. Hardware schedules themselves are deterministic.
        with log.open('w') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'RAYON_NUM_THREADS': '1'})
        for line in log.read_text().splitlines():
            if 'invariants=PASS' not in line:
                continue
            fields = dict(token.split('=', 1) for token in shlex.split(line) if '=' in token)
            rows.append({'phase': int(fields['phase']), 'configuration': name, 'family': family,
                         'cycles': int(fields['horizonCycles']),
                         'maximum_row_bytes': 4*int(fields['maximumRowWords']),
                         'total_row_bytes': 4*int(fields['rowWords']),
                         'first_activity_span_cycles': int(fields['firstActivitySpanCycles']),
                         'endpoint_lower_bound_cycles': int(fields['endpointLowerBoundCycles']),
                         'schedule_ms': float(fields['scheduleCodegenMedianMs']),
                         'transfers': int(fields['transfers']), 'destinations': int(fields['destinations']),
                         'row_fingerprint': fields['rowFingerprint']})
        (args.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (args.output/'manifest.json').write_text(json.dumps({'revision': revision, 'commands': commands,
        'rayon_threads': 1, 'iterations': 1, 'snapshot': str(args.snapshot.resolve())}, indent=2)+'\n')
    render(rows, args.output)


if __name__ == '__main__':
    main()
