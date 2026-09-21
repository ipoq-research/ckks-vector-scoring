# Release audit: IPOQ CKKS Vector Scoring Research v0.1.0

**Prepared:** 21 September 2026. **Disposition at audit completion:** local research release candidate ready for owner review. This report records the engineering audit completed before GitHub upload. Later documentation and packaging changes are recorded in [repository setup notes](REPOSITORY_SETUP.md).

## Scope and independence

This is a focused engineering review of source provenance, reproduction, numerical correctness, evidence consistency, packaged contents and licensing notices. It was performed within the same AI-assisted project workflow. It is **not an independent audit, cryptographic security certification, legal title opinion, novelty determination or production-readiness approval**.

The review covers the documented fresh public-query path and the associated client/import tests. Secondary legacy profiling and direction-reuse helpers are present for provenance and imports; their performance is not a release claim. Upstream cryptographic implementations and dependency archives were not comprehensively re-audited.

## Evidence and execution

Native dependencies were compiled from their bundled, checksum-pinned archives into a new build tree. Existing pinned Python installations were reused. The host was a Linux x86-64 environment exposing an Intel Xeon Platinum 8573C and AVX-512 F/DQ/IFMA. The build used GCC 13.3.0, CMake 4.1.3, Python 3.12.14, NumPy 2.3.5 and seal-python 4.4.0. No GPU was available for this release audit.

| Check | Result | Evidence |
| --- | --- | --- |
| Clean native dependency and bridge compilation | Passed for CPU; both retained and current clients compiled | `evidence/audit/provenance.json` |
| Differential ciphertext import | 2,828 cases: 846 accepted with checked decryption, 1,982 rejected | `evidence/audit/import_validation_cpu.json` |
| Bounded workload corners | 80 checks; max observed absolute error `3.3213e-7` | Same import record |
| Alternate parameter/scale fallback | 8 checks; max observed absolute error `2.1892e-6` | Same import record |
| Exact CRT52 SIMD reconstruction and conversion | 327,680 checks and 3 invalid-input rejections | `evidence/audit/crt52_cpu.json` |
| Fresh CPU pipeline benchmark | 5 fresh keys, 80 exact ciphertext comparisons, 80 exact modular-decryption checks, full numerical checks | `evidence/audit/packed_cpu.json` |
| Full-score fresh CPU challenge | 65,536 scores checked; 16 ciphertext parity checks, 16 exact decryption checks, 3 malformed replies rejected | `evidence/audit/cpu_receipt.json` |
| Separate scalar score recomputation | All 65,536 scores passed; max recomputed error `1.35191e-9` | `evidence/audit/evidence_check.json` |
| Timing reconstruction | Archived GPU, fresh CPU and separate receipt statistics recomputed from raw samples | Same evidence-check record |
| Evidence tampering regressions | 5 tests passed, including incorrect scores with a recomputed envelope hash | `tests/test_evidence.py` |

The checked tolerance was `1e-5`. Corner/fallback error values belong to different tests than the five-key scoring headline. None is asserted to be a worst-case analytical error bound.

Timing results:

| Run | Encrypted median | Fastest included plaintext | Ratio | Relative 10× met? |
| --- | ---: | ---: | ---: | --- |
| Archived GPU, five keys | 260.175 µs | 26.4075 µs | 9.8523× | Yes, in the archived record |
| Fresh CPU, five keys | 579.5525 µs | 26.2330 µs | 22.0925× | No |
| Fresh CPU, separate one-key receipt | 640.5435 µs | 27.1335 µs | 23.6071× | No |

The complete raw-sample records remain in the release. CPU timing and correctness testing ran separately to avoid deliberately benchmarking while the validation suite was active. Shared-host scheduling and system noise remain uncontrolled. There was no dedicated bare-metal isolation, CPU-frequency locking, host attestation or independent operator.

## Claims corrected or constrained

1. **General FHE and exact-search wording:** replaced with a specialized bounded CKKS scoring workload, approximate arithmetic, public queries, prepared encrypted records and all-score output. No ANN indexing, arbitrary-depth execution or general-purpose compiler result is claimed.
2. **Mixed client phase figures:** the final archived GPU experiment's separately measured client median is `157.8165 µs`. The earlier `142.74 µs` value was from a different optimization test and is not combined with the final pipeline. Separate phase medians are not added to manufacture a full-pipeline median.
3. **Moving plaintext denominator:** the archived 9.8523× ratio uses `26.4075 µs` from that run. The earlier absolute target `234.93 µs` remains unmet by `260.175 µs`.
4. **Independent proof wording:** numerical parity, raw-sample reconstruction, independent scalar arithmetic and external reproduction are distinguished. No external reproduction has been established.
5. **Source inventory omission:** archived source hashes omit `.cu`. New inventories include it. All 18 included engine files that are present in the historical inventory match their original pre-release hashes. The CUDA source and requirements file are outside that historical inventory. The missing historical CUDA digest cannot be repaired retroactively.
6. **Receipt interpretation:** a receipt SHA-256 is an integrity checksum, not a signature or a proof of encrypted execution. The new checker recalculates statistics; tamper tests verify failures for specific corruptions. A sufficiently fabricated internally consistent record is outside its detection scope.
7. **Key isolation:** the native fresh-query API takes no secret key, but the same-process Python fixture contains keys and plaintext. No host/process isolation result is claimed.
8. **License scope:** original code and documents receive separate explicit licenses. Third-party source and notices retain their existing terms. The optimized source is included intentionally.

## Source and licensing review

The 20 engine files have retained their native arithmetic and existing benchmark behavior. Changes consist of copyright/SPDX comments and inclusion of `.cu` files in the source digest inventory. A byte-identity check after reversing those documented changes matched all 20 original engine hashes. The challenge wrapper was adapted to the release layout and research wording. New build helpers, evidence checkers, tests and documentation are separately visible in the package.

`evidence/audit/original_source_hashes.json` records the imported source snapshot. `provenance.json` records original wrapper digests, historical evidence digests, the source match inventory, build environment and release changes. Historical JSON records are copied unchanged. Archived source inventories also mention a few older scripts outside this deliberately smaller release; the documented fresh-query reproduction path has its required files.

Bundled SEAL, HEXL, cpu_features and PocketFFT digests were checked against their pinned metadata. SEAL's MIT license and complete nested NOTICE, HEXL and cpu_features Apache-2.0 licenses, and PocketFFT's BSD-3-Clause notice are retained. Full Apache-2.0 and CC BY 4.0 texts and a material-to-license map are included. Python wheels, the CUDA toolkit and drivers are external prerequisites, not redistributed artifacts.

The review confirms notice presence and the declared mapping. It does not independently establish ownership of every contribution, upstream provenance beyond the recorded sources, absence of patent claims, or supply-chain trust. SHA-256 consistency does not provide a signed chain of custody.

## Packaging review

The archive is assembled from an explicit allowlist of release source, pinned vendor files, documentation, tests, notices and evidence. Expanded dependency build trees, compiled objects/shared libraries, virtual environments, cloud administration utilities, browser profiles, billing data, presenter controls, private keys and unrelated research files are excluded.

Automated checks inspect the final selected first-party text files for common token/private-key patterns, credential-bearing URLs and leaked workspace paths; confirm no binary/key filenames are selected; inspect vendor archive paths; validate dependency digests; parse Python source; and resolve local Markdown links. Original vendor archives necessarily contain upstream authors' public contact details and their own test/build material. They are not rewritten or represented as IPOQ-authored content.

`scripts/package_release.py` implements the checks and produces `SHA256SUMS`. After extracting the candidate archive into a new directory, the offline evidence checker and tamper regression tests are run against the extracted files. `evidence/audit/package_checks.json` records the packaging checks. Pattern scans are limited checks, not a guarantee that every possible secret format is detectable.

## Remaining limits

- GPU compilation and execution were not repeated during this audit; the CUDA-specific branches rely on historical evidence. CPU compilation of the shared source does not test CUDA execution.
- No matched EVA, OPERA/HOMCACHE or Cachemir evaluation, comprehensive literature review, or first-in-world claim.
- No real query traces, production dataset evaluation, concurrent service load, remote networking measurement or deployment isolation test.
- No systematic sanitizer/fuzzing campaign, formal correctness proof, malicious-ciphertext security analysis, side-channel review or independent security audit.
- Python dependencies are version-pinned but not wheel-hash-locked; compiled binaries are machine-specific. Fresh cryptographic randomness and system load prevent exact numerical/timing replay.
- Publication and access to the intended `ipoq-research` GitHub destination remain separate from the local release checks.

## Reproduce this review

Follow the root README to install the Python requirements and build, then run:

```bash
bash scripts/validate.sh
python engine/benchmark_packed.py --output results/packed_cpu_reproduction.json
python run_experiment.py --backend cpu --seed 123 --query-seed 456
python scripts/audit_evidence.py
```

New executions write new results; they do not overwrite the archived release evidence. The documented build, full tests and benchmark are available for another operator to run on a different machine. That would provide the external reproduction this release currently lacks.
