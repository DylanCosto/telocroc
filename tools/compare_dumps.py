#!/usr/bin/env python3
"""Compare telocroc debug dumps ($TELOCROC_DUMP) with Telogator2's temp/ files.

usage: compare_dumps.py PY_OUT/temp RS_DUMP_DIR
"""
import glob, os, sys
import numpy as np

py_temp, rs_dump = sys.argv[1], sys.argv[2]
n_ok = n_bad = n_missing = 0
for npz in sorted(glob.glob(os.path.join(py_temp, '*', 'npz', '*.npz')) + glob.glob(os.path.join(py_temp, '03_collapse', '*.npz'))):
    rel = os.path.relpath(npz, py_temp).replace('/npz/', '/')[:-4]
    f32 = os.path.join(rs_dump, rel + '.f32')
    py = np.load(npz)['dist']
    if not os.path.exists(f32):
        n_missing += 1
        print('missing', rel, py.shape)
        continue
    rs = np.fromfile(f32, dtype='<f4')
    if rs.size != py.size:
        n_bad += 1
        print('SHAPE', rel, py.shape, rs.size)
        continue
    rs = rs.reshape(py.shape)
    diff = np.abs(rs.astype(np.float64) - py.astype(np.float64))
    if np.array_equal(rs, py):
        n_ok += 1
    else:
        n_bad += 1
        print('DIFF', rel, py.shape, 'cells', int((diff > 0).sum()), 'max', float(diff.max()))
print(f'matrices: {n_ok} identical, {n_bad} different, {n_missing} missing')

def read_fa(fn):
    out, name = {}, None
    for line in open(fn):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]; out[name] = ''
        elif name:
            out[name] += line
    return out

c_ok = c_bad = 0
for fa in sorted(glob.glob(os.path.join(py_temp, '*', 'fa', '*.fa'))):
    rel = os.path.relpath(fa, py_temp).replace('/fa/', '/')
    mine = os.path.join(rs_dump, rel)
    if not os.path.exists(mine):
        print('missing', rel); continue
    a, b = read_fa(fa), read_fa(mine)
    if a == b:
        c_ok += 1
    else:
        c_bad += 1
        for k in a:
            if a.get(k) != b.get(k):
                x, y = a.get(k, ''), b.get(k, '')
                first = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
                print('CONS', rel, k, 'len', len(x), len(y), 'first diff at', first)
print(f'consensus files: {c_ok} identical, {c_bad} different')
