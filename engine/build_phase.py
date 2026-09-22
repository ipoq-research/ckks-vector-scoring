# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Compile both retained and experimental clients with identical flags/dependencies."""
import json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
vendor=ROOT/'vendor'
meta=json.loads((vendor/'matched_source.json').read_text())
source=vendor/meta['source_dir'];build=vendor/'hexl_seal_build';prefix=vendor/'accelerated_install'
if not (build/'lib').exists():raise RuntimeError('Run python build_native.py --hexl first')
commands=[]
for name,out in [('client_retained.cpp','client_retained.so'),('client_native.cpp','client_native_hexl.so')]:
 command=['c++','-O3','-march=native','-std=c++17','-fPIC','-shared','-fvisibility=hidden','-Wl,-Bsymbolic',
 '-I'+str(source/'native/src'),'-I'+str(build/'native/src'),'-I'+str(prefix/'include'),str(ROOT/name),
 str(next((build/'lib').glob('libseal-*.a'))),str(prefix/'lib/libhexl.a'),str(prefix/'lib/libcpu_features.a'),'-pthread','-o',str(ROOT/out)]
 subprocess.run(command,check=True);commands.append(command)
(ROOT/'results/build_phase.json').write_text(json.dumps({'commands':commands,'compiler':subprocess.check_output(['c++','--version'],text=True),'sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['client_retained.cpp','client_native.cpp']}},indent=2)+'\n')
