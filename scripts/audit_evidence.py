"""Recompute published timing statistics and numerical receipt checks, offline.

This checks the consistency of supplied evidence, not its authenticity.
"""
# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_receipt import verify


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    check(type(actual) in (int, float) and math.isfinite(actual), label + ': nonfinite value')
    check(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label + ': statistic mismatch')


def block_median(block, count, query_count, unit):
    samples = block['samples_' + unit]
    check(len(samples) == count and count > 0, 'Wrong sample count')
    check(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples), 'Invalid timing sample')
    check(sorted(block['query_order']) == list(range(query_count)), 'Invalid query permutation')
    value = statistics.median(samples) * (1e6 if unit == 's' else 1)
    close(block['median_us'], value, 'Block median')
    return value


def benchmark(record):
    config = record['configuration']
    trials = record['trials']
    check(len(trials) == config['trials'] and len(trials) > 0, 'Incomplete trials')
    names = set(record['summary_us'])
    required = {'pipeline_packed', 'pipeline_standard', 'server_packed', 'server_standard', 'client_mode10', 'plaintext_cpu', 'plaintext_blas'}
    check(names == required | ({'plaintext_gpu'} if config['gpu'] else set()), 'Unexpected methods')
    collected = {name: [] for name in names}
    for index, trial in enumerate(trials):
        check(trial['trial'] == index and trial['fresh_key'] is True, 'Invalid key trial metadata')
        check(trial['parameters'] == [50, 50, 56] and trial['security'] == 'SEAL tc128 accepted', 'Unexpected parameter metadata')
        check(trial['response_bytes'] == 393329, 'Unexpected response size')
        check(trial['exact_server_byte_checks'] == trial['exact_decryption_checks'] == config['queries'], 'Incomplete reference checks')
        check(0 <= trial['max_error'] < 1e-5, 'Error exceeds declared tolerance')
        check(trial['invalid_queries_rejected'] == 4 and trial['output_ownership_checked'] is True, 'Missing rejection/ownership checks')
        check(len(trial['method_orders']) == config['blocks'], 'Wrong method order count')
        check(all(len(order) == len(names) and set(order) == names for order in trial['method_orders']), 'Invalid method permutation')
        check(set(trial['measurements']) == names, 'Inconsistent method sets')
        for name in names:
            entry = trial['measurements'][name]
            check(len(entry['blocks']) == config['blocks'] and config['blocks'] > 0, 'Incomplete blocks')
            value = statistics.median(block_median(b, config['samples'], config['queries'], 's') for b in entry['blocks'])
            close(entry['median_us'], value, 'Trial median')
            collected[name].append(value)
    medians = {name: statistics.median(values) for name, values in collected.items()}
    for name in names:
        close(record['summary_us'][name], medians[name], 'Aggregate median')
    best = min((n for n in names if n.startswith('plaintext_')), key=medians.get)
    check(record['fastest_plaintext'] == best, 'Wrong plaintext comparator')
    ratio = medians['pipeline_packed'] / medians[best]
    close(record['overhead'], ratio, 'Overhead')
    check(record['all_passed'] is True, 'Failed benchmark')
    return {'trials': len(trials), 'pipeline_us': medians['pipeline_packed'], 'fastest_plaintext': best,
            'plaintext_us': medians[best], 'overhead': ratio,
            'max_error': max(t['max_error'] for t in trials),
            'pipeline_samples': len(trials) * config['blocks'] * config['samples']}


def receipt_timing(receipt):
    timing = receipt['payload']['timing']
    names = set(timing['methods'])
    check({'encrypted_pipeline', 'plaintext_native_cpu', 'plaintext_blas'} <= names, 'Missing controls')
    check(len(timing['method_orders']) == timing['blocks'], 'Missing method orders')
    check(all(len(order) == len(names) and set(order) == names for order in timing['method_orders']), 'Invalid method order')
    medians = {}
    for name, blocks in timing['methods'].items():
        check(len(blocks) == timing['blocks'], 'Missing receipt blocks')
        medians[name] = statistics.median(block_median(b, timing['samples_per_method_per_block'], timing['queries'], 'us') for b in blocks)
    check(set(timing['medians_us']) == names, 'Missing summary methods')
    for name, value in medians.items():
        close(timing['medians_us'][name], value, 'Receipt median')
    best = min((n for n in names if n.startswith('plaintext_')), key=medians.get)
    ratio = medians['encrypted_pipeline'] / medians[best]
    check(best == timing['fastest_plaintext'], 'Receipt comparator mismatch')
    close(timing['overhead'], ratio, 'Receipt overhead')
    check(timing['median_10x_met'] == (ratio <= 10), 'Incorrect relative threshold flag')
    check(timing['absolute_234_93_us_met'] == (medians['encrypted_pipeline'] <= 234.93), 'Incorrect absolute threshold flag')
    return {'medians_us': medians, 'overhead': ratio}


def manifest(root):
    entries = (root / 'SHA256SUMS').read_text().splitlines()
    seen = set()
    for line in entries:
        digest, name = line.split('  ', 1)
        path = root / name
        check(not Path(name).is_absolute() and '..' not in Path(name).parts and name not in seen, 'Unsafe or duplicate manifest path')
        check(path.is_file() and not path.is_symlink(), 'Manifest file missing or symlinked: ' + name)
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'File hash mismatch: ' + name)
        seen.add(name)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    results = {'scope': 'Offline consistency checks; no authenticated timing or independent reproduction.', 'benchmarks': {}}
    for relative in ['evidence/historical/packed_gpu.json', 'evidence/audit/packed_cpu.json']:
        results['benchmarks'][relative] = benchmark(json.loads((root / relative).read_text()))
    receipt_path = root / 'evidence/audit/cpu_receipt.json'
    results['receipt_numerical'] = verify(receipt_path)
    results['receipt_timing'] = receipt_timing(json.loads(receipt_path.read_text()))
    if (root / 'SHA256SUMS').exists():
        results['manifest_files_verified'] = manifest(root)
    results['passed'] = True
    rendered = json.dumps(results, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as exc:
        print('AUDIT FAILED: ' + str(exc), file=sys.stderr)
        sys.exit(1)
