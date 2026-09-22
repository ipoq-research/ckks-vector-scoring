# What It Takes to Benchmark Encrypted Vector Scoring Honestly

**IPOQ Research | v0.1.0 | 21 September 2026 | CC BY 4.0**

We are releasing the source, evidence and reproduction tools from a specialized encrypted vector-scoring experiment. The archived GPU run reported a complete local request in **260.175 microseconds**, or **9.8523 times** its fastest tested plaintext control. Preparing this release produced a fresh CPU result of **579.5525 microseconds and 22.0925 times plaintext**.

Those numbers describe different machines and execution paths. Sharing both makes the result useful: readers can see what was measured, inspect the implementation and run the experiment themselves.

## The question we actually tested

Our starting ambition was encrypted computation with overhead close to ordinary computation. We narrowed the experiment to a concrete operation: compute the squared Euclidean distance from a public 32-dimensional query to each of 4,096 encrypted records.

For one record, the operation is:

\[
d(x,q)=\lVert x\rVert^2-2\langle x,q\rangle+\lVert q\rVert^2.
\]

The database is prepared before online requests. We encrypt feature columns so that a ciphertext packs one feature across all 4,096 records, and precompute the encrypted squared norms. Each public query then supplies scalar coefficients to the encrypted dot product. The server returns a full encrypted score vector; the client decrypts and decodes all 4,096 values.

This structure avoids rotations for the feature sum because the corresponding record occupies the same slot in each feature ciphertext. It also keeps query-independent work outside the online request. The plaintext controls receive the corresponding preparation benefit: transposed records and precomputed squared norms.

The experiment uses approximate CKKS arithmetic. Parameters are accepted by SEAL at `tc128`, with ring dimension 8,192, input scale `2^46` and modulus bit sizes `[50,50,56]`. The tested input bounds are `[-1,1]` for records and `[-0.5,0.5]` for queries. Every score must satisfy an absolute tolerance of `1e-5`.

The release covers a bounded scoring workload. Extending its performance result to arbitrary programs, private queries, ANN indexing or production search would require additional implementation and evidence.

## Where the useful changes came from

The work used standard CKKS and existing arithmetic libraries. The engineering changes concentrated on specialization and data movement.

The server evaluates the public-query arithmetic directly over ciphertext residues in NTT representation. A shared source file provides a native CPU path and a CUDA path. Its reply serializer reuses storage for the complete wire representation, then returns an independently owned immutable byte string. It still transfers every ciphertext component.

The native client combines validated import with modular decryption and specialized decoding. Eligible moduli and CPU instruction support enable AVX-512 IFMA arithmetic. The CRT path reconstructs and centers integers before conversion to floating point. A half-size FFT and slot mapping recover the real outputs. Where the optimized shape does not apply, the client uses supported fallback paths.

These changes do not eliminate validation. The fast importer requires the expected serialized header, exact payload size and canonical residues. Other supported layouts pass through SEAL's normal validation path. The release contains differential tests against a retained client and exact modular-decryption comparisons against SEAL.

One modest but measurable result is particularly easy to inspect: in the archived GPU run, changing reply serialization reduced the complete local median from **281.5225 to 260.175 microseconds**, about **7.58%**, under the same benchmark configuration. That is a systems improvement in a specific pipeline.

## A timing boundary readers can inspect

The benchmark includes public-query preparation, server evaluation, complete reply construction, synchronous GPU transfers when applicable, validated client import, decryption, decoding and an owned output array. It excludes key generation, database encryption and preparation, initial context creation and network transport.

That last exclusion matters. The reply alone is **393,329 bytes**. Local pipeline latency cannot be presented as a remote service's end-to-end latency. Query transport, network bandwidth, queuing, isolation and service overhead require separate measurements.

Five-key experiments use five fresh key generations, 16 rotating queries, three method blocks per key and 64 requests per block. Method and query orders are randomized and retained. We calculate a median within each block, a median across blocks for each key, and a median across keys. The published ratio divides aggregate encrypted and plaintext medians. We preserve every recorded timing sample instead of presenting only the best run.

| Run | Local encrypted median | Fastest tested plaintext median | Ratio |
| --- | ---: | ---: | ---: |
| Archived RTX A6000 / Xeon Gold 6342 | 260.175 µs | 26.4075 µs | 9.8523× |
| Release audit, CPU on Xeon Platinum 8573C | 579.5525 µs | 26.2330 µs | 22.0925× |

The GPU's relative 10× threshold was met against its same-run control. An earlier absolute target of **234.93 microseconds was not met**. The fastest control is the fastest implementation included in that run, not a proof that no better plaintext implementation exists.

Individual phase medians also need care. The final archived run measured its client separately at **157.8165 microseconds**. An earlier **142.74-microsecond** client measurement came from a different test and is not a component of the final reported run. Summing independently measured phase medians does not reconstruct the full-pipeline median.

## Correctness, provenance and what remains open

The fresh release build passed **2,828 differential import cases**, comprising 846 accepted cases with checked decryption and 1,982 rejected cases. It also passed 80 workload corner checks, eight parameter/scale fallback checks, and **327,680 exact CRT reconstruction and conversion checks**. The five-key CPU benchmark performed 80 full ciphertext byte comparisons and 80 exact modular-decryption comparisons, alongside all-score numerical checks.

The archived GPU and fresh five-key CPU scoring runs observed maximum absolute errors of approximately `1.13e-9` and `1.76e-9`. Broader rescaled and fallback tests have larger errors, still below the declared tolerance. Observed maxima should not be interpreted as a theorem about every input or parameter choice.

We also provide a separate fresh CPU challenge with dataset seed 123 and query seed 456. It records all **65,536 returned scores** for the selected query and 15 companion queries. A verifier using only Python's standard library regenerates the public data and recomputes each distance with scalar arithmetic. The separate challenge measured 640.5435 microseconds and 23.6071 times plaintext; it is not substituted into the five-key benchmark.

These checks serve different purposes. Exact ciphertext parity tests implementation agreement with SEAL. Scalar recalculation checks numerical outputs. Raw-sample reconstruction checks reported statistics. None authenticates a remote machine, proves that an untrusted party executed encryption, or establishes the whole application's security.

The native server interface accepts no secret key, but our benchmark deliberately places plaintext, keys and both roles in one process. It tests arithmetic and local timing, not deployment isolation. Ciphertext validation does not authenticate a well-formed altered ciphertext or prove correct server evaluation.

Historical evidence also has a limitation: its source inventory omitted the CUDA file's hash. The source is included in this release, and new source inventories cover `.cu` files, but that change cannot retroactively attest to the historical GPU binary. No GPU was available for the release audit. Independent GPU reproduction, deployment testing and a cryptographic review remain open.

We have not completed matched comparisons with EVA, OPERA/HOMCACHE or Cachemir. This article therefore makes no priority or world-first claim. Those comparisons need equivalent workloads, confidentiality assumptions, parameter choices, preparation costs and output boundaries.

## Why publish it

The useful contribution of this release is an inspectable experiment: source, pinned native dependencies, measurements, failure checks, documented boundaries and a route to reproduce the result. It shows how specializing arithmetic, reducing allocation and measuring the complete client phase can improve a particular encrypted workload.

We are publishing it so others can test the result on different hardware, improve the controls, find defects and reuse the code. The original implementation is Apache-2.0; this article and original evidence are CC BY 4.0. Upstream components retain their own licenses.

Start with the repository's `README.md`, run `python3 scripts/audit_evidence.py` to inspect the supplied evidence, then build and execute the CPU or GPU experiment. Please report the hardware, full timing boundary, raw samples and unsuccessful results along with any improvement.

## Sources and evidence

All numerical claims above derive from the included [archived GPU record](../evidence/historical/packed_gpu.json), [fresh CPU benchmark](../evidence/audit/packed_cpu.json), [fresh import validation](../evidence/audit/import_validation_cpu.json), [CRT checks](../evidence/audit/crt52_cpu.json), and [full-score receipt](../evidence/audit/cpu_receipt.json). The [release audit](RELEASE_AUDIT.md) records verification and limitations.

The implementation builds on [Microsoft SEAL](https://github.com/microsoft/SEAL), [Intel HEXL](https://github.com/intel/hexl), [Google cpu_features](https://github.com/google/cpu_features), and [PocketFFT](https://github.com/mreineck/pocketfft). See the release's [third-party notices](../THIRD_PARTY_NOTICES.md) for pinned revisions and license files.
