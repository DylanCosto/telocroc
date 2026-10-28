#!/usr/bin/env python3
"""Compare two tlens_by_allele.tsv files allele by allele.

usage: compare_tsv.py A.tsv B.tsv

Alleles are matched by their supporting reads (the read sets overlap most), so the
comparison works even when allele ids or order differ between runs.
"""
import sys


def load(fn):
    rows = []
    for line in open(fn):
        if line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        rows.append({'chr': f[0], 'pos': f[1], 'ref': f[2], 'id': f[3], 'tl': int(f[4]),
                     'reads': set(f[10].split(',')), 'tvr_len': int(f[8]), 'line': f})
    return rows


a, b = load(sys.argv[1]), load(sys.argv[2])
used = set()
identical = same_reads = matched = 0
tl_diffs = []
unmatched_a = []
for ra in a:
    best, best_j = 0.0, None
    for j, rb in enumerate(b):
        if j in used:
            continue
        jac = len(ra['reads'] & rb['reads']) / len(ra['reads'] | rb['reads'])
        if jac > best:
            best, best_j = jac, j
    if best_j is None or best < 0.5:
        unmatched_a.append(ra)
        continue
    used.add(best_j)
    rb = b[best_j]
    matched += 1
    if ra['line'][:4] + ra['line'][4:] == rb['line'][:4] + rb['line'][4:] or ra['line'][:3] + ra['line'][4:] == rb['line'][:3] + rb['line'][4:]:
        identical += 1
    if ra['reads'] == rb['reads']:
        same_reads += 1
    tl_diffs.append((abs(ra['tl'] - rb['tl']), ra['chr'], ra['pos'], ra['tl'], rb['tl'], ra['tvr_len'], rb['tvr_len'], (ra['chr'], ra['pos']) == (rb['chr'], rb['pos'])))
unmatched_b = [rb for j, rb in enumerate(b) if j not in used]

print(f'{len(a)} alleles in A, {len(b)} in B; {matched} matched by supporting reads')
print(f'  identical rows (ignoring allele id): {identical}')
print(f'  same supporting reads: {same_reads}')
print(f'  same chr/position: {sum(1 for d in tl_diffs if d[7])}')
diffs = sorted(d[0] for d in tl_diffs)
if diffs:
    print(f'  |TL difference|: median {diffs[len(diffs) // 2]}, max {diffs[-1]}, '
          f'zero for {sum(1 for d in diffs if d == 0)}, <=50bp for {sum(1 for d in diffs if d <= 50)}')
for d in sorted(tl_diffs, reverse=True)[:8]:
    if d[0]:
        print(f'    {d[1]}:{d[2]} TL {d[3]} vs {d[4]} (tvr_len {d[5]} vs {d[6]})')
for name, rows in (('A', unmatched_a), ('B', unmatched_b)):
    for r in rows:
        print(f'  only in {name}: {r["chr"]}:{r["pos"]} id {r["id"]} TL {r["tl"]} ({len(r["reads"])} reads)')
