# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import argparse,json,subprocess
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cuda',action='store_true');args=ap.parse_args()
 prefix=ROOT/'vendor'/'accelerated_install';libs=[prefix/'lib/libhexl.a',prefix/'lib/libcpu_features.a']
 if not all(p.exists() for p in libs):raise RuntimeError('Build the native client with --hexl first')
 if args.cuda:
  cmd=['nvcc','-O3','-std=c++17','-arch=sm_80','-Xcompiler','-fPIC','-Xcompiler','-pthread','-shared','-DFHE_HEXL','-I'+str(prefix/'include'),str(ROOT/'fresh_backend.cu'),*map(str,libs),'-o',str(ROOT/'fresh_cuda.so')]
 else:
  cmd=['c++','-x','c++','-O3','-march=native','-std=c++17','-fPIC','-shared','-DFHE_HEXL','-I'+str(prefix/'include'),str(ROOT/'fresh_backend.cu'),'-x','none',*map(str,libs),'-pthread','-o',str(ROOT/'fresh_cpu.so')]
 subprocess.run(cmd,check=True);(ROOT/'results'/('cuda_build.json' if args.cuda else 'cpu_build.json')).write_text(json.dumps({'command':cmd},indent=2)+'\n')
if __name__=='__main__':main()
