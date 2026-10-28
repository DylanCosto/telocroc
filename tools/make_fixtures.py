#!/usr/bin/env python3
"""Regenerate tests/fixtures/python_reference.json from the real Python libraries."""
import json, random, sys
sys.path.insert(0, sys.argv[1])  # upstream Telogator2 checkout
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from source.tg_kmer import get_nonoverlapping_kmer_hits, read_kmer_tsv, wavelet_smooth
from source.tg_util import RC

rng = random.Random(5)
out = {}
out['wavelet'] = []
for n in (113, 257, 1001, 2048):
    x = [rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]) - rng.choice([0.0, 0.5, 1.0]) for _ in range(n)]
    y = wavelet_smooth(np.array(x))
    out['wavelet'].append({'x': x, 'y': [float(v) for v in y]})
out['percentile'] = []
for _ in range(20):
    v = [rng.randint(-2000, 20000) for _ in range(rng.randint(1, 30))]
    q = rng.choice([50, 75, 90, 33])
    out['percentile'].append({'v': v, 'q': q, 'p': float(np.percentile(v, q))})
out['linkage'] = []
for method in ('ward', 'single'):
    for _ in range(6):
        n = rng.randint(3, 40)
        pts = [[rng.random(), rng.random()] for _ in range(n)]
        d = [float(np.float32(np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]))) for i in range(n) for j in range(i + 1, n)]
        z = linkage(np.array(d), method=method)
        if np.max(z[:, 2]) > 1:
            z[:, 2] /= np.max(z[:, 2])
        for cut in (0.05, 0.2, 0.5):
            labels = fcluster(z, cut, 'distance').tolist()
            groups = {}
            for i, l in enumerate(labels):
                groups.setdefault(l, []).append(i)
            out['linkage'].append({'method': method, 'n': n, 'd': d, 'cut': cut, 'clusters': sorted(groups.values())})
meta, issub, canon = read_kmer_tsv(sys.argv[1] + '/source/resources/kmers.tsv', 'hifi')
kmers = meta[0]
out['kmer_hits'] = []
units = ['CCCTAA'] * 12 + ['CCCTGA', 'CCCTCA', 'CCCCAA', 'CCTA', 'CCCAA', 'CCCTAAA', 'CCGTGA']
for _ in range(10):
    seq = ''.join(rng.choice(units) if rng.random() < 0.9 else rng.choice('ACGT') for _ in range(400))
    hits = get_nonoverlapping_kmer_hits(seq, [RC(k) for k in kmers], issub)
    out['kmer_hits'].append({'seq': seq, 'hits': [[list(s) for s in h] for h in hits]})
json.dump(out, open(sys.argv[2], 'w'))
print('wrote', sys.argv[2])
