# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Shareable numerical verifier. Standard library only; contains no FHE implementation."""
import argparse, hashlib, json, math, struct, sys
from pathlib import Path

ROWS, COLS, TOLERANCE = 4096, 32, 1e-5
DOMAIN = b'IPOQ-DEMO-DATA-v1\0'

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()

def data_values(seed):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Dataset seed must be an unsigned 64-bit integer')
    raw = hashlib.shake_256(DOMAIN + seed.to_bytes(8, 'big')).digest(ROWS * COLS * 4)
    return [(u - 2**31) / 2**31 for (u,) in struct.iter_unpack('<I', raw)]

def data_digest(values):
    return hashlib.sha256(struct.pack('<%dd' % len(values), *values)).hexdigest()

def query_for_seed(seed, index=0):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Query seed must be an unsigned 64-bit integer')
    raw = hashlib.shake_256(b'IPOQ-DEMO-QUERY-v1\0' + seed.to_bytes(8, 'big') + index.to_bytes(4, 'big')).digest(COLS*4)
    return [((u & ((1 << 20)-1)) - 2**19) / 2**20 for (u,) in struct.iter_unpack('<I', raw)]

def check(condition, message):
    if not condition:
        raise ValueError(message)

def verify(path):
    path = Path(path)
    check(path.stat().st_size <= 12 * 1024 * 1024, 'Receipt too large')
    receipt = json.loads(path.read_text())
    p = receipt['payload']
    check(receipt['sha256'] == hashlib.sha256(canonical(p)).hexdigest(), 'Receipt digest mismatch')
    check(p['schema'] == 'ipoq-demo-receipt-v1', 'Unknown receipt schema')
    check(p['shape'] == [ROWS, COLS], 'Unexpected workload shape')
    check(p['declared_tolerance'] == TOLERANCE, 'Unexpected tolerance')
    values = data_values(p['dataset_seed'])
    check(data_digest(values) == p['dataset_sha256'], 'Dataset commitment mismatch')
    records = [values[i*COLS:(i+1)*COLS] for i in range(ROWS)]
    queries = p['queries']
    check(len(queries) == 16, 'Expected the selected query plus 15 companion queries')
    worst, count = 0., 0
    query_for_seed(p['query_seed'])
    for index, entry in enumerate(queries):
        check(entry['index'] == index, 'Unexpected query index')
        q, scores = entry['query'], entry['decrypted_scores']
        check(len(q) == COLS and all(type(v) in (float, int) and math.isfinite(v) and abs(v) <= .5 for v in q), 'Invalid query')
        if index:
            check(q == query_for_seed(p['query_seed'], index), 'Companion query differs from committed seed')
        check(len(scores) == ROWS and all(type(v) in (float, int) and math.isfinite(v) for v in scores), 'Invalid output vector')
        # Independent scalar fsum calculation; does not trust stored reference scores/errors.
        for row, actual in zip(records, scores):
            expected = math.fsum((x-y)*(x-y) for x, y in zip(row, q))
            error = abs(expected-actual)
            check(error <= TOLERANCE, 'Numerical mismatch exceeds fixed tolerance')
            worst = max(worst, error)
            count += 1
    return {'passed': True, 'scores_checked': count, 'maximum_recomputed_error': worst,
            'scope': 'Checks numerical results and receipt consistency only. Does not independently establish encrypted execution, security, timing, or author identity.'}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('receipt', type=Path)
    args = ap.parse_args()
    try:
        result = verify(args.receipt)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as e:
        print('VERIFICATION FAILED: ' + str(e), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
