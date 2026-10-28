#!/usr/bin/env python3
"""Replay telocroc MSA dumps through Telogator2's own tg_align functions."""
import glob, sys
import numpy as np
sys.path.insert(0, sys.argv[1])  # upstream checkout
from source.tg_align import get_aligner_object, get_scoring_matrix, progressive_alignment, iterative_refinement, get_final_tvr_consensus, pairwise_align

canon = 'C'
msa_aligner = get_aligner_object(scoring_matrix=get_scoring_matrix(canon, 'msa'), gap_bool=(True, False), which_type='msa')
ref_aligner = get_aligner_object(scoring_matrix=get_scoring_matrix(canon, 'msa_refinement'), gap_bool=(True, False), which_type='msa')
dub = sys.argv[3] if len(sys.argv) > 3 else 'AVWY'
stats = {'prog': 0, 'refine': 0, 'cons': 0, 'n': 0}
for fn in sorted(glob.glob(sys.argv[2] + '/msa_*.txt')):
    d = {'seq': [], 'dist': [], 'initial': [], 'refined': [], 'cons': []}
    for line in open(fn):
        tag, val = line.rstrip('\n').split('\t')
        d[tag].append(val)
    seqs = d['seq']
    dists = np.array([[np.float32(x) for x in r.split(',')] for r in d['dist']], dtype=np.float32)
    stats['n'] += 1
    prog = progressive_alignment(seqs, dists, msa_aligner)
    ok_p = list(prog) == d['initial']
    stats['prog'] += ok_p
    ref = iterative_refinement(d['initial'], ref_aligner)
    ok_r = list(ref) == d['refined']
    stats['refine'] += ok_r
    cons = get_final_tvr_consensus(d['refined'], default_char=canon, untrustworthy_chars=[canon] + list(dub), tiebreak_adj={canon: 1, 'A': -1})
    ok_c = cons == d['cons'][0]
    stats['cons'] += ok_c
    print(fn.split('/')[-1], 'n', len(seqs), 'lens', [len(s) for s in seqs][:3], 'prog', ok_p, 'refine', ok_r, 'cons', ok_c, flush=True)
    if not ok_p and len(seqs) >= 3:
        # find the first differing row
        for i, (a, b) in enumerate(zip(prog, d['initial'])):
            if a != b:
                first = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
                print('   row', i, 'len', len(a), len(b), 'first diff', first)
                break
print(stats)
