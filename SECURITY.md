# Research security scope

This is an experimental scoring implementation and benchmark, not a production cryptographic service. The release review is an engineering audit by the same AI-assisted project workflow. It is not an independent security audit or certification.

The tested operation evaluates squared Euclidean distances for 4,096 encrypted records with 32 real features and a **public query**. Records are bounded by `[-1,1]`; query coordinates by `[-0.5,0.5]`. CKKS arithmetic is approximate. The native client requires parameter acceptance at SEAL's `tc128` setting. Acceptance is a parameter check, not proof of the whole application's security.

The native fresh-query server API receives ciphertext residues and public parameters, not a secret key. The benchmark harness deliberately holds plaintext, keys and both roles in one Python process to check correctness. Its fixture object contains the key, so the Python wrapper does not establish process or host isolation. A malicious host running this harness can access all data.

Clients and backends reuse internal scratch buffers and are not safe for concurrent calls to one instance. The C ABI assumes trusted in-process callers and valid pointer/length pairs. Do not expose it directly to arbitrary foreign pointers or concurrent callers.

Malformed ciphertext serialization is checked before the specialized decode path. The differential import tests are finite checks, not exhaustive fuzzing or a memory-safety proof. Well-formed ciphertext alterations are not authenticated. This release does not implement verifiable computation, malicious-server resistance, a decryption-oracle defense, side-channel hardening, authenticated transport, tenant separation, query privacy or access control.

Full scores are intentionally returned to the key holder. Authorization and inference risks from repeated score queries are outside this experiment. No claim of circuit privacy or protection from the recipient of decrypted results is made.

The code has not undergone a systematic sanitizer campaign or an independent cryptographic review. Do not interpret successful reference comparisons as establishing semantic security, correctness for every possible input, or resistance to implementation attacks. No secret keys or real customer data are included in the release.

For research defects, provide a minimal synthetic reproducer in the eventual research repository. A private vulnerability-reporting channel has not yet been established; do not put sensitive keys or user data in public issues.
