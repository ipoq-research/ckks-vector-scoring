# IPOQ CKKS Vector Scoring Research

**v0.1.0 | Research source release | 21 September 2026**

A reproducible experiment in evaluating **4,096 squared Euclidean distances over 32 encrypted real features per record, using a public query**. It combines a prepared CKKS database, native CPU or optional CUDA evaluation, validated ciphertext import, and native client decryption and decoding.

Start with the [research article](docs/what-it-takes-to-benchmark-encrypted-vector-scoring-honestly.md), [release audit](docs/RELEASE_AUDIT.md), and [security scope](SECURITY.md). Source repository: [ipoq-research/ckks-vector-scoring](https://github.com/ipoq-research/ckks-vector-scoring). See [repository setup notes](docs/REPOSITORY_SETUP.md) for the relationship to the audited archive and the [IPOQ release announcement](https://ipoq.com/blog/introducing-ckks-vector-scoring/) for an introduction.

## Results and their provenance

| Experiment | Hardware | Full local pipeline | Fastest tested plaintext | Ratio |
| --- | --- | ---: | ---: | ---: |
| Archived GPU experiment, five fresh-key trials | RTX A6000 + Xeon Gold 6342 | 260.175 µs | 26.4075 µs, CUDA | 9.8523× |
| Fresh release CPU experiment, five fresh-key trials | Xeon Platinum 8573C | 579.5525 µs | 26.2330 µs, NumPy BLAS | 22.0925× |
| Separate fresh CPU receipt, dataset seed 123 / query seed 456 | Xeon Platinum 8573C | 640.5435 µs | 27.1335 µs, NumPy BLAS | 23.6071× |

Each five-key result is the median of per-key medians of three block medians, with 64 requests per block and 16 rotating queries. The ratio divides aggregate encrypted and aggregate plaintext medians. It is not a latency guarantee or a measurement of general-purpose FHE overhead. The one-key receipt is a separate run with a different dataset.

The archived GPU run met a relative 10× median threshold against its contemporaneous control. It did **not** meet the earlier absolute 234.93 µs target. No GPU was available for this release audit; CUDA execution and the historical GPU timing were not rerun. Independent reproduction remains outstanding. Maximum observed errors in the five-key GPU and CPU experiments were `1.1321e-9` and `1.7621e-9`, respectively. Those observed values are not worst-case guarantees.

## What is measured

The timed request begins with an available public query and a prepared encrypted database. It includes server query preparation, evaluation, complete serialized reply construction, device transfers when using CUDA, validated client import, decryption, full decoding, and an independently owned array of 4,096 results.

Key generation, encryption, database preparation, native context initialization and network transport are excluded. The receipt records its setup time separately. Every reply includes the complete ciphertext, including the reusable nonlinear component. The benchmark does not time a cached answer or move distance calculation to the client.

The workload uses CKKS with ring dimension 8,192, input scale `2^46`, and modulus bit sizes `[50,50,56]`, accepted by SEAL at `tc128`. Outputs use scale `2^92`; an uncompressed three-component reply has **393,329 bytes**. Inputs are synthetic records in `[-1,1]`, with public query coordinates in `[-0.5,0.5]`. Arithmetic is approximate with a checked absolute tolerance of `1e-5`. This release implements scoring, not an ANN index, top-k retrieval, private queries, arbitrary-depth evaluation or a general compiler.

## Check the supplied evidence without compiling

Python's standard library is sufficient:

```bash
python3 scripts/audit_evidence.py
python3 -m unittest discover -s tests -v
python3 verify_receipt.py evidence/audit/cpu_receipt.json
```

The audit recomputes medians and ratios from raw samples, checks counts and permutations, verifies the release file hashes when `SHA256SUMS` is present, and recalculates all 65,536 scores in the fresh receipt with scalar arithmetic. Negative tests reject a forged headline, a dropped timing sample, a fabricated median, incorrect scores with a valid envelope hash, and a replaced companion query. A hash or self-generated receipt is not an authenticated execution proof.

## Build and execute the experiment

Tested on Linux x86-64 with Python 3.12, GCC 13.3, NumPy 2.3.5 and seal-python 4.4.0. Install Python development/runtime support, a C++17 compiler, and `make` through your platform. Use an isolated Python environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r engine/requirements.txt
bash scripts/build.sh
bash scripts/validate.sh
python engine/benchmark_packed.py --output results/packed_cpu.json
python run_experiment.py --backend cpu --seed 123 --query-seed 456
```

`engine/benchmark_packed.py` interprets a relative output path under `engine/`; receipt outputs go under root `results/`. It defaults to five trials, 16 queries, three blocks and 64 samples. Key generation and encryption use fresh randomness, so ciphertexts, numerical noise and timings will differ between executions. Native dependency source is bundled; installing Python packages requires access to a package index. No binary-reproducible build is claimed.

Do not set `PYTHONOPTIMIZE` or run with `-O`; the research harness uses assertions. Native compilation uses `-march=native`, so rebuild on each target CPU instead of copying shared libraries. `IPOQ_PYTHON` can select the Python executable for the shell scripts. `FHE_CMAKE` can select a CMake executable. No root privileges or secrets are required by the experiment itself.

The optimized client selects eligible SIMD paths at runtime. The CPU experiment can use supported fallbacks; `validate.sh` explicitly skips the dedicated CRT52 tests if AVX-512 F/DQ/IFMA is unavailable. Such a skip is not a passing SIMD test.

### Optional GPU execution

The supplied CUDA build targets `sm_80` and requires a compatible NVIDIA GPU, driver and CUDA toolkit. The documented full GPU challenge also requires a CPU exposing AVX-512 F/DQ/IFMA for the optimized client.

```bash
bash scripts/build.sh --cuda
python engine/benchmark_packed.py --gpu --output results/packed_gpu_reproduction.json
python run_experiment.py --backend gpu --seed 123 --query-seed 456
```

GPU timings include synchronous host/device transfers for the online request. Freshly report both encrypted and plaintext results on the same machine. Do not substitute the archived 9.85× ratio for a new measurement.

## Layout

| Path | Purpose |
| --- | --- |
| `engine/client_native.cpp` | Validated import, modular decryption, CRT and FFT decoding |
| `engine/fresh_backend.cu` | CPU and optional CUDA public-query evaluation and plaintext controls |
| `engine/packed_backend.py` | Full serialization into reusable storage, followed by an owned bytes reply |
| `engine/client_retained.cpp` | Retained client for differential tests |
| `engine/benchmark_packed.py` | Matched final pipeline benchmark |
| `engine/verify_import.py`, `engine/verify_crt52.py` | Import, corner, fallback and exact integer validation |
| `run_experiment.py`, `verify_receipt.py` | Fresh challenge and separate numerical receipt verification |
| `evidence/historical/` | Unmodified archived GPU evidence |
| `evidence/audit/` | Fresh CPU results, full score receipt, audit and provenance records |
| `engine/vendor/`, `licenses/` | Pinned native sources and preserved upstream notices |

Secondary profiling and direction-reuse helpers are retained because the original harness imports them and they document earlier experiments. This release's measured claim concerns the fresh public-query scoring path. It does not establish the benefit of direction caching on real query traces.

## License and reuse

Original code is **Apache-2.0**; original article, documentation and benchmark evidence are **CC BY 4.0**, to the extent applicable rights exist. Third-party material keeps its original licenses. See [LICENSING.md](LICENSING.md), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This open release includes the optimized C++/CUDA implementation and permits commercial reuse under the stated terms.

## Project credit

Project leadership and research direction: **Erik Groset**. Research publishing entity: **IPOQ, LLC**, using the IPOQ project brand. This work used substantial AI assistance for implementation, experiments, analysis and writing. See [AUTHORS.md](AUTHORS.md) for roles, upstream credit and ownership qualifications, and [CITATION.cff](CITATION.cff) for citation metadata. Project credit does not establish sole authorship, patent inventorship or exclusive legal ownership.

## Commercial use

The licenses permit commercial products and services subject to their terms. Paid integration, support, deployment engineering and future product development are possible business models for IPOQ and other users. No separate IPOQ license fee or contribution-back obligation is imposed on permitted use of this release. The research package itself includes no service-level commitment, independent security certification or production-readiness guarantee. See [LICENSING.md](LICENSING.md) for patent-grant and redistribution considerations.

## Contact

For research questions, independent reproduction or integration discussions, contact **Erik Groset, IPOQ, LLC**, at <erik@ipoq.com>. See [SECURITY.md](SECURITY.md) for reporting security concerns.
