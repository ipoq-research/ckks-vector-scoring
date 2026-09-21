# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Randomized matched ablations, fresh keys, identical data/query/output boundaries."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,ctypes,json,statistics,subprocess,time
from pathlib import Path
import numpy as np
import seal
from profile_client import fixture,measure
from native_client import NativeClient
from reproduce import machine,rotating
from seal_reproduction import binary
from profile_phases import PHASES
ROOT=Path(__file__).resolve().parent
CONFIGS={'original':(56,46,56),'ifma':(50,50,56)}
def configure(c):
 c.lib.client_profile_current.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];c.lib.client_profile_current.restype=ctypes.c_int
 c.lib.client_verify_fast.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];c.lib.client_verify_fast.restype=ctypes.c_int
 c.lib.client_import_stats.argtypes=[ctypes.c_void_p,ctypes.c_void_p];c.lib.client_import_stats.restype=ctypes.c_int
def verify(c,b):
 assert c.lib.client_verify_fast(c.handle,b,len(b)),c.lib.client_error()
def stats(c):
 v=(ctypes.c_uint64*2)();assert c.lib.client_import_stats(c.handle,v);return {'fast':v[0],'fallback':v[1]}
def main():
 if not __debug__:raise RuntimeError('Assertions required')
 ap=argparse.ArgumentParser();ap.add_argument('--trials',type=int,default=5);ap.add_argument('--queries',type=int,default=16);ap.add_argument('--blocks',type=int,default=3);ap.add_argument('--samples',type=int,default=64);ap.add_argument('--gpu',action='store_true');ap.add_argument('--output',default='results/import_local.json');args=ap.parse_args()
 assert min(args.trials,args.queries,args.blocks,args.samples)>0
 r={'machine':machine(),'configuration':vars(args),'scope':'Fresh public queries against encrypted 4096x32 database. Online server reply plus CPU client import, decrypt, full decode and owned output. Setup/network excluded; no external reproduction.','trials':[]}
 if args.gpu:
  from fresh_backend import FreshBackend,PlainGPU
  r['machine']['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
 for trial in range(args.trials):
  rng=np.random.default_rng(604701+trial);x=rng.uniform(-1,1,(4096,32));qs=[np.ascontiguousarray(rng.integers(-2**19,2**19,32)/2**20) for _ in range(args.queries)]
  targets=[np.sum((x-q)**2,axis=1) for q in qs];methods={};resources=[];configs={};phase_profiles={};validation=[]
  for label,bits in CONFIGS.items():
   t=time.perf_counter();f=fixture(604701+trial,bits,x);setup=time.perf_counter()-t;s=f['server'];blobs=[binary(s.reply(s.fresh(q),q,False)) for q in qs]
   t=time.perf_counter();c=NativeClient(f['params'],f['keys'].secret_key(),True);client_setup=time.perf_counter()-t;old=NativeClient(f['params'],f['keys'].secret_key(),True,'client_retained.so');configure(c);resources.extend([c,old])
   configs[label]={'bits':bits,'moduli':[p.value() for p in f['context'].key_context_data().parms().coeff_modulus()],'security':'SEAL tc128 accepted','data_bits':f['context'].first_context_data().total_coeff_modulus_bit_count(),'key_bits':f['context'].key_context_data().total_coeff_modulus_bit_count(),'fixture_setup_s':setup,'client_setup_s':client_setup,'response_bytes':len(blobs[0])}
   inputs=list(zip(blobs,targets))
   for _ in range(4):
    a=rng.uniform(-1,1,4096);b=rng.uniform(-1,1,4096);e=seal.Encryptor(f['context'],f['keys'].create_public_key());ct=s.eval.multiply(e.encrypt(f['encoder'].encode(a,2.**46)),e.encrypt(f['encoder'].encode(b,2.**46)));inputs.append((binary(ct),a*b))
   for b,target in inputs:
    ct=seal.Ciphertext();ct.load_bytes(f['context'],b)
    for level in [ct,s.eval.rescale_to_next(ct)]:
     data=binary(level);verify(c,data);ref=f['encoder'].decode(f['decryptor'].decrypt(level));previous=old.decode(data,8)
     for m in [8,9,10]:
      y=c.decode(data,m);err=float(np.max(np.abs(y-target)));diff=float(np.max(np.abs(y-ref)));delta=float(np.max(np.abs(y-previous)))
      assert err<1e-5 and diff<1e-9 and delta<1e-9,(label,m,err,diff,delta)
      validation.append({'config':label,'mode':m,'limbs':level.coeff_modulus_size(),'error':err,'seal_difference':diff,'retained_difference':delta})
   for n,cl,m in [('retained8',old,8),('mode8',c,8),('mode9',c,9),('mode10',c,10)]:
    methods[label+'_client_'+n]=[lambda b=b,cl=cl,m=m:cl.decode(b,m) for b in blobs]
   if args.gpu:
    t=time.perf_counter();g=FreshBackend(f,True);resources.append(g);configs[label]['gpu_setup_s']=time.perf_counter()-t
    for q,b,target in zip(qs,blobs,targets):
     assert g.reply(q)==b
     assert np.max(np.abs(c.decode(g.reply(q),10)-target))<1e-5
    methods[label+'_gpu_server']=[lambda q=q,g=g:g.reply(q) for q in qs]
    for n,cl,m in [('retained8',old,8),('mode9',c,9),('mode10',c,10)]:methods[label+'_pipeline_'+n]=[lambda q=q,g=g,cl=cl,m=m:cl.decode(g.reply(q),m) for q in qs]
    if label=='original':
     cpu=FreshBackend(f);resources.append(cpu);xx=np.ascontiguousarray(x.T);norm=np.ascontiguousarray(np.sum(x*x,axis=1));pg=PlainGPU(g.lib,xx,norm);resources.append(pg)
     def pc(q):
      out=np.empty(4096);assert cpu.lib.fresh_plain(xx.ctypes.data,norm.ctypes.data,q.ctypes.data,out.ctypes.data);return out
     def blas(q):return norm-2*(q@xx)+q@q
     for q,target in zip(qs,targets):
      for fn in [pc,pg.reply,blas]:assert np.max(np.abs(fn(q)-target))<1e-10
     methods['plaintext_gpu']=[lambda q=q:pg.reply(q) for q in qs];methods['plaintext_cpu']=[lambda q=q:pc(q) for q in qs];methods['plaintext_blas']=[lambda q=q:blas(q) for q in qs]
   configs[label]['import_stats_after_validation']=stats(c)
   phase_profiles[label]=(c,blobs)
  blocks={n:[] for n in methods};orders=[]
  for _ in range(args.blocks):
   names=list(methods);rng.shuffle(names);orders.append(names)
   for n in names:
    order=rng.permutation(args.queries).tolist();blocks[n].append(measure(rotating(methods[n],order),args.samples)|{'query_order':order})
  pr={};out=np.empty(4096);phases=np.empty(8)
  for label,(c,blobs) in phase_profiles.items():
   for m in [8,9,10]:
    ps=[]
    for i in range(args.samples):
     b=blobs[i%len(blobs)];assert c.lib.client_profile_current(c.handle,b,len(b),m,out.ctypes.data,phases.ctypes.data),c.lib.client_error();assert np.max(np.abs(out-c.decode(b,m)))<1e-9;ps.append(phases.tolist())
    pr[label+'_mode'+str(m)]={'samples_us':ps,'median_us':dict(zip(PHASES,np.median(ps,axis=0).tolist()))}
  ms={n:{'median_us':statistics.median(b['median_us'] for b in bs),'blocks':bs} for n,bs in blocks.items()}
  row={'trial':trial,'fresh_keys_per_config':True,'parameters':configs,'validation':validation,'exact_decryption_checks':2*(args.queries+4)*len(CONFIGS),'server_byte_checks':args.queries*len(CONFIGS) if args.gpu else 0,'measurements':ms,'method_orders':orders,'phase_profiles':pr}
  r['trials'].append(row);r['summary_us']={n:statistics.median(t['measurements'][n]['median_us'] for t in r['trials']) for n in methods};r['phase_summary_us']={n:{p:statistics.median(t['phase_profiles'][n]['median_us'][p] for t in r['trials']) for p in PHASES} for n in pr};r['all_checks_passed']=True
  if args.gpu:
   r['fastest_plaintext']=min((n for n in methods if n.startswith('plaintext')),key=lambda n:r['summary_us'][n]);r['overhead']=r['summary_us']['ifma_pipeline_mode10']/r['summary_us'][r['fastest_plaintext']]
  (ROOT/args.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'trial':trial,'summary':r['summary_us'],'overhead':r.get('overhead')},indent=2),flush=True)
  for obj in reversed(resources):obj.close()
if __name__=='__main__':main()
