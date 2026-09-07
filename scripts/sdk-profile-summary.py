#!/usr/bin/env python3
"""Extract annotated SDK profile evidence; run with the SDK Python and pva enabled."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3
import pva


def stage(name):
    for pattern, label in [('/attn/Softmax/', 'score layout / softmax'),
                           ('/attn/MatMul_1', 'PV'), ('/attn/MatMul/', 'QK transpose'),
                           ('/attn/qkv/', 'fused QKV projection'),
                           ('/attn/proj/', 'output projection'),
                           ('/attn/', 'Q/K scaling or conversion'),
                           ('/mlp/fc1/', 'MLP up'), ('/mlp/fc2/', 'MLP down'),
                           ('/mlp/', 'MLP activation')]:
        if pattern in name:
            return label
    return 'other'


def summarize(path, output):
    db = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    report = pva.openReport(str(path))
    programs = {p._id: p for p in report.compilation.programs}
    sets = {row['compute_set']: report.compilation.computeSets[row['compute_set']]
            for row in db.execute('select compute_set from compute_sets')}
    rows = []
    for r in db.execute('''select s.id,s.type,s.program,s.compute_set,p.name program_name,
            c.name compute_name,i.cycles,i.active_tiles,i.cycles_base,
            i.active_cycles_from_min,i.active_cycles_to_max
            from steps s join steps_by_ipu i on s.id=i.step_id
            left join programs p on s.program=p.id
            left join compute_sets c on s.compute_set=c.compute_set order by s.id'''):
        name = r['compute_name'] or r['program_name'] or ''
        p = programs.get(r['program'])
        cs = sets.get(r['compute_set'])
        rows.append(dict(step=r['id'], kind=r['type'], stage=stage(name), name=name,
                         cycles=r['cycles'], active_tiles=r['active_tiles'],
                         start=r['cycles_base'] + r['active_cycles_from_min'],
                         end=r['cycles_base'] + r['active_cycles_to_max'],
                         max_exchange_code_bytes=max(p.codeBytesByTile)
                         if p is not None and p.type == pva.Program.Type.DoExchange else 0,
                         vertices='; '.join(v.type.name for v in cs.vertices) if cs else ''))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'steps.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    qkv = [r for r in rows if r['name'].endswith('/Convolve') and '/attn/qkv/' in r['name']]
    first = [r for r in rows if qkv[0]['step'] <= r['step'] < qkv[1]['step']]
    with (output / 'first-body.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(first)
    kernels = [r for r in first if r['name'].endswith('/Convolve')]
    exchanges = [dict(name=p.name, maximum=max(p.codeBytesByTile),
                      mean=sum(p.codeBytesByTile) / len(p.codeBytesByTile))
                 for p in programs.values() if p.type == pva.Program.Type.DoExchange
                 and '/blocks.0/attn/' in p.name]
    memory = dict(db.execute('''select min(b),avg(b),max(b) from (
        select sum(v.size_bytes) b from vars_info v join var_categories c on c.id=v.category
        where c.name='internalExchangeCode' group by tile)''').fetchone())
    # Exclude host waits. Measure from the first device exchange after input
    # delivery to the last compute completion before the output stream begins.
    spans = []
    for incoming in [r for r in rows if r['kind'] == 'StreamCopyEnd' and r['name'] == 'copyFromHost']:
        following = [r for r in rows if r['step'] > incoming['step']]
        outgoing = next(r for r in following if r['kind'] == 'StreamCopyBegin' and r['name'].endswith('copyToHost'))
        body = [r for r in following if r['step'] < outgoing['step'] and r['kind'] in ['DoExchange', 'OnTileExecute']]
        spans.append(max(r['end'] for r in body) - min(r['start'] for r in body))
    result = dict(profile=str(path), kernels=kernels, attention_exchanges=exchanges,
                  exchange_memory_per_tile=memory, device_spans=spans,
                  qkv_executions=len(qkv),
                  allocated_memory=dict(db.execute('select min(total),avg(total),max(total) from memory_by_tile').fetchone()))
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(path, 'device spans:', spans, 'exchange bytes:', memory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profiles', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/sdk-profile-analysis'))
    args = parser.parse_args()
    for path in args.profiles:
        summarize(path, args.output / path.parent.parent.name)
