# Third-party components

Native dependency archives are bundled unchanged and checked by SHA-256 before first extraction. Their JSON records in `engine/vendor/` contain source URLs, pinned revisions, archive names and digests. These hashes verify consistency with the recorded artifacts; they are not upstream signatures.

| Component | Pinned source | License and notice location |
| --- | --- | --- |
| Microsoft SEAL, matching SEAL-Python v4.4.0 | `04d53b99ce745efc26bb4965be609b9894755227` | MIT: `licenses/matched_LICENSE.txt`; full nested notices: `licenses/matched_NOTICE.txt`; originals inside archive |
| Intel HEXL | `75a60b8cc908fb4c166315dd8b55b2e1ee7faea7` | Apache-2.0: `licenses/hexl_LICENSE.txt`; originals inside archive |
| Google cpu_features | `32b49eb5e7809052a28422cfde2f2745fbb0eb76` | Apache-2.0: `licenses/cpu_features_LICENSE.txt`; originals inside archive |
| PocketFFT | `c90e55b3d529f8efa40ed01a20de22405f45fc65` | BSD-3-Clause: `licenses/PocketFFT-BSD-3-Clause.txt`, `engine/vendor/pocketfft_LICENSE.md`, and header notices |

Microsoft SEAL's retained NOTICE identifies additional upstream material, including material under MIT, BSD, Apache and CC0 terms. Preserve that complete notice when redistributing the included SEAL source. Original IPOQ source imports and uses SEAL internals; retained upstream notices apply to any adaptations as well.

The Python environment is installed separately. `engine/requirements.txt` pins NumPy 2.3.5 (BSD-3-Clause, with additional notices for packaged dependencies), seal-python 4.4.0 (MIT), and CMake 4.1.3 (BSD-3-Clause and included component notices). Their wheels are not redistributed here. Consult and preserve their installed license files if redistributing an environment. The Python pins are not a hash-locked wheel environment.

The optional NVIDIA CUDA toolkit and driver are external prerequisites and are not redistributed or relicensed. No compiled binaries, container images, BLAS libraries or GPU drivers are shipped.

Primary upstream locations: [SEAL](https://github.com/microsoft/SEAL), [SEAL-Python](https://github.com/Huelse/SEAL-Python), [HEXL](https://github.com/intel/hexl), [cpu_features](https://github.com/google/cpu_features), [PocketFFT](https://github.com/mreineck/pocketfft).
