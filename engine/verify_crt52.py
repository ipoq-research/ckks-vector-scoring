# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Exact SIMD integer reconstruction and conversion tests, including carry/rounding boundaries."""
import ctypes,json
from pathlib import Path
import numpy as np
import seal
ROOT=Path(__file__).resolve().parent
def main():
 if not __debug__:raise RuntimeError('Assertions required')
 lib=ctypes.CDLL(str(ROOT/'client_native_hexl.so'));lib.client_error.restype=ctypes.c_char_p
 lib.client_verify_crt52.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint64,ctypes.c_uint64];lib.client_verify_crt52.restype=ctypes.c_int
 rng=np.random.default_rng(207851);records=[]
 for bits in [[56,46],[50,50],[52,51],[59,44],[40,40]]:
  primes=seal.CoeffModulus.Create(8192,bits);p0,p1=[p.value() for p in primes];Q=p0*p1
  cases=[0,1,p0-1,p0,p0+1,Q//2-1,Q//2,Q//2+1,Q-2,Q-1]
  for e in range(1,Q.bit_length()):
   for delta in [-1,0,1]:
    cases.append((1<<e)+delta)
    if e>53:cases.append((1<<e)+(1<<(e-53))+delta)
  cases=[z for z in cases if 0<=z<Q];a=rng.integers(0,p0,8192,dtype=np.uint64);t=rng.integers(0,p1,8192,dtype=np.uint64)
  for i,z in enumerate(cases):a[i]=z%p0;t[i]=z//p0
  for rep in range(8):
   if rep:a=rng.integers(0,p0,8192,dtype=np.uint64);t=rng.integers(0,p1,8192,dtype=np.uint64)
   assert lib.client_verify_crt52(a.ctypes.data,t.ctypes.data,8192,p0,p1),lib.client_error().decode()
  records.append({'bits':bits,'p0':p0,'p1':p1,'exact_integer_and_conversion_checks':8*8192,'boundary_cases':len(cases)})
 # Noncanonical inputs and an unsupported range must be refused.
 a[0]=p0
 assert not lib.client_verify_crt52(a.ctypes.data,t.ctypes.data,8192,p0,p1)
 assert not lib.client_verify_crt52(a.ctypes.data,t.ctypes.data,8191,p0,p1)
 assert not lib.client_verify_crt52(a.ctypes.data,t.ctypes.data,8192,p0,1<<53)
 d={'records':records,'all_passed':True,'invalid_input_rejections':3,'total_exact_checks':sum(r['exact_integer_and_conversion_checks'] for r in records)}
 (ROOT/'results/crt52_validation.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
