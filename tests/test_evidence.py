"""Ensure evidence checkers detect plausible tampering even after rehashing."""
# SPDX-License-Identifier: Apache-2.0
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from audit_evidence import benchmark, receipt_timing
from verify_receipt import canonical, verify


class EvidenceChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((ROOT / 'evidence/historical/packed_gpu.json').read_text())
        cls.receipt = json.loads((ROOT / 'evidence/audit/cpu_receipt.json').read_text())

    def test_forged_headline(self):
        bad = copy.deepcopy(self.record)
        bad['overhead'] = 1.0
        with self.assertRaises(ValueError):
            benchmark(bad)

    def test_dropped_slow_sample(self):
        bad = copy.deepcopy(self.record)
        samples = bad['trials'][0]['measurements']['pipeline_packed']['blocks'][0]['samples_s']
        samples.remove(max(samples))
        with self.assertRaises(ValueError):
            benchmark(bad)

    def test_invented_receipt_median(self):
        bad = copy.deepcopy(self.receipt)
        bad['payload']['timing']['medians_us']['encrypted_pipeline'] = 0.1
        with self.assertRaises(ValueError):
            receipt_timing(bad)

    def test_incorrect_scores_despite_valid_hash(self):
        bad = copy.deepcopy(self.receipt)
        bad['payload']['queries'][0]['decrypted_scores'][0] += 0.01
        bad['sha256'] = hashlib.sha256(canonical(bad['payload'])).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            path.write_bytes(canonical(bad))
            with self.assertRaisesRegex(ValueError, 'Numerical mismatch'):
                verify(path)

    def test_replaced_companion_query_despite_valid_hash(self):
        bad = copy.deepcopy(self.receipt)
        bad['payload']['queries'][1] = copy.deepcopy(bad['payload']['queries'][0])
        bad['payload']['queries'][1]['index'] = 1
        bad['sha256'] = hashlib.sha256(canonical(bad['payload'])).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'receipt.json'
            path.write_bytes(canonical(bad))
            with self.assertRaisesRegex(ValueError, 'Companion query'):
                verify(path)
