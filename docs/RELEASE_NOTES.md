# v0.1.0 release notes

Initial open research package for the specialized IPOQ CKKS vector-scoring experiment. This repository snapshot contains the audited v0.1.0 implementation, evidence and licensing framework. Repository setup details are recorded in `REPOSITORY_SETUP.md`.

Included: native C++ client, CPU/CUDA public-query server source, Python benchmark and challenge harnesses, pinned native dependency sources, historical GPU measurements, fresh CPU validation, full-score receipt, offline evidence checker, tamper regression tests, article and audit report.

Original code: Apache-2.0. Original documentation and evidence: CC BY 4.0. Upstream material retains its notices and licenses.

Archived GPU result: 260.175 µs / 9.8523× same-run plaintext. Fresh CPU result: 579.5525 µs / 22.0925× same-run plaintext. These are bounded local scoring experiments with preparation and networking excluded. Independent GPU reproduction and security review remain outstanding.

Repository: `ipoq-research/ckks-vector-scoring`. A future research article on `ipoq.com` can link to this repository.
