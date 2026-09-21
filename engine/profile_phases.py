# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Measure the current half-FFT client; preserve ordinary request timing separately."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,ctypes,json,statistics,time
from pathlib import Path
import numpy as np
from profile_client import fixture,measure
from native_client import NativeClient
from reproduce import machine,rotating
from seal_reproduction import binary
ROOT=Path(__file__).resolve().parent
PHASES=['validated_import','modular_decryption','plaintext_validation','coefficient_copy','inverse_ntt','crt_conversion_twist','half_fft','slot_gather']
def main():
 if not __debug__:raise RuntimeError('Assertions required')
 ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=3);ap.add_argument('--queries',type=int,default=8);ap.add_argument('--samples',type=int,default=32);ap.add_argument('--blocks',type=int,default=3);args=ap.parse_args()
 record={'machine':machine(),'configuration':vars(args),'scope':'CPU client only on current host, fresh public-query replies. Timers attribute a single current decode path; ordinary wrapper timings are measured separately. Import includes SEAL copying and validation. Decoder coefficient copy is separate. No GPU/network timings inferred.','trials':[]}
 for trial in range(args.trials):
  f=fixture(307041+trial);s=f['server'];rng=f['rng'];queries=[rng.integers(-2**19,2**19,32)/2**20 for _ in range(args.queries)]
  blobs=[binary(s.reply(s.fresh(q),q,False)) for q in queries];targets=[np.sum((f['x']-q)**2,axis=1) for q in queries]
  c=NativeClient(f['params'],f['keys'].secret_key(),True);old=NativeClient(f['params'],f['keys'].secret_key(),True,'client_retained.so')
  c.lib.client_profile_current.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];c.lib.client_profile_current.restype=ctypes.c_int
  output=np.empty(4096);phases=np.empty(8)
  measurements={};errors=[];phase_records={}
  for mode in [4,6]:
   for b,target in zip(blobs,targets):
    assert c.lib.client_profile_current(c.handle,b,len(b),mode,output.ctypes.data,phases.ctypes.data)
    assert np.array_equal(output,c.decode(b,mode));assert np.array_equal(output,old.decode(b,mode));err=float(np.max(np.abs(output-target)));assert err<1e-5;errors.append(err)
   block_records=[]
   for block in range(args.blocks):
    values=[];outer=[]
    for j in range(5):c.lib.client_profile_current(c.handle,blobs[j%len(blobs)],len(blobs[j%len(blobs)]),mode,output.ctypes.data,phases.ctypes.data)
    for i in range(args.samples):
     b=blobs[int(rng.integers(len(blobs)))];start=time.perf_counter_ns()
     assert c.lib.client_profile_current(c.handle,b,len(b),mode,output.ctypes.data,phases.ctypes.data)
     outer.append((time.perf_counter_ns()-start)/1000);values.append(phases.tolist())
    block_records.append({'phase_samples_us':values,'outer_samples_us':outer,'phase_medians_us':dict(zip(PHASES,np.median(values,axis=0).tolist()))})
   phase_records[str(mode)]=block_records
  methods={f'{label}_mode{mode}':[lambda b=b,c=cl,m=mode:c.decode(b,m) for b in blobs] for label,cl in [('retained',old),('instrumented_source',c)] for mode in [4,6]}
  for name in methods:measurements[name]=[]
  for _ in range(args.blocks):
   names=list(methods);rng.shuffle(names)
   for name in names:
    order=rng.permutation(len(blobs)).tolist();measurements[name].append(measure(rotating(methods[name],order),args.samples)|{'query_order':order})
  record['trials'].append({'trial':trial,'fresh_key':True,'max_error':max(errors),'phase_records':phase_records,'ordinary_measurements':measurements})
  record['phase_summary_us']={str(mode):{p:statistics.median(statistics.median(b['phase_medians_us'][p] for b in t['phase_records'][str(mode)]) for t in record['trials']) for p in PHASES} for mode in [4,6]}
  record['ordinary_summary_us']={name:statistics.median(statistics.median(b['median_us'] for b in t['ordinary_measurements'][name]) for t in record['trials']) for name in methods};record['all_checks_passed']=True
  (ROOT/'results/phases.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'trial':trial,'phases':record['phase_summary_us'],'ordinary':record['ordinary_summary_us']},indent=2),flush=True)
  c.close();old.close()
if __name__=='__main__':main()
