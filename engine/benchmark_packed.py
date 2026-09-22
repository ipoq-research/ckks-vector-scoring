# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Matched end-to-end serialization change, identical crypto and output ownership."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,statistics,subprocess
from pathlib import Path
import numpy as np
from profile_client import fixture,measure
from packed_backend import PackedBackend
from fresh_backend import PlainGPU
from native_client import NativeClient
from reproduce import machine,rotating
from seal_reproduction import binary
from benchmark_import import configure,verify
ROOT=Path(__file__).resolve().parent
def main():
 if not __debug__:raise RuntimeError('Assertions required')
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',action='store_true');ap.add_argument('--trials',type=int,default=5);ap.add_argument('--queries',type=int,default=16);ap.add_argument('--blocks',type=int,default=3);ap.add_argument('--samples',type=int,default=64);ap.add_argument('--output',default='results/packed_gpu.json');args=ap.parse_args()
 assert min(args.trials,args.queries,args.blocks,args.samples)>0
 r={'machine':machine(),'configuration':vars(args),'scope':'Same full local fresh-query pipeline; both wrappers return complete, independently owned serialized bytes. No network or setup included. Fresh keys, full 4096-value output.','trials':[]}
 if args.gpu:r['machine']['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True)
 for trial in range(args.trials):
  f=fixture(604701+trial,(50,50,56));rng=f['rng'];qs=[np.ascontiguousarray(rng.integers(-2**19,2**19,32)/2**20) for _ in range(args.queries)];x=f['x'];s=f['server'];g=PackedBackend(f,args.gpu);cpu=g if not args.gpu else PackedBackend(f);c=NativeClient(f['params'],f['keys'].secret_key(),True);configure(c)
  targets=[np.sum((x-q)**2,axis=1) for q in qs];expected=[binary(s.reply(s.fresh(q),q,False)) for q in qs];checks=[];saved=[]
  for q,b,target in zip(qs,expected,targets):
   packed=g.reply_packed(q);assert packed==g.reply(q)==b;assert isinstance(packed,bytes);saved.append(packed);verify(c,packed);y=c.decode(packed,10);err=float(np.max(np.abs(y-target)));assert err<1e-5;checks.append(err)
  assert saved==expected,'Previously returned reply was modified'
  for bad in [np.full(32,np.nan),np.full(32,np.inf),np.full(32,.5001),np.zeros(31)]:
   try:g.reply_packed(bad)
   except ValueError:pass
   else:raise AssertionError('Bad query accepted')
  assert g.reply_packed(qs[0])==expected[0]
  xx=np.ascontiguousarray(x.T);norm=np.ascontiguousarray(np.sum(x*x,axis=1))
  def pc(q):
   out=np.empty(4096);assert cpu.lib.fresh_plain(xx.ctypes.data,norm.ctypes.data,q.ctypes.data,out.ctypes.data);return out
  def blas(q):return norm-2*(q@xx)+q@q
  methods={'server_standard':[lambda q=q:g.reply(q) for q in qs],'server_packed':[lambda q=q:g.reply_packed(q) for q in qs],
   'pipeline_standard':[lambda q=q:c.decode(g.reply(q),10) for q in qs],'pipeline_packed':[lambda q=q:c.decode(g.reply_packed(q),10) for q in qs],
   'client_mode10':[lambda b=b:c.decode(b,10) for b in expected],
   'plaintext_cpu':[lambda q=q:pc(q) for q in qs],'plaintext_blas':[lambda q=q:blas(q) for q in qs]}
  for q,target in zip(qs,targets):
   for fn in [pc,blas]:assert np.max(np.abs(fn(q)-target))<1e-10
  if args.gpu:
   pg=PlainGPU(g.lib,xx,norm);methods['plaintext_gpu']=[lambda q=q:pg.reply(q) for q in qs]
   for q,target in zip(qs,targets):assert np.max(np.abs(pg.reply(q)-target))<1e-10
  blocks={n:[] for n in methods};orders=[]
  for _ in range(args.blocks):
   names=list(methods);rng.shuffle(names);orders.append(names)
   for n in names:
    order=rng.permutation(args.queries).tolist();blocks[n].append(measure(rotating(methods[n],order),args.samples)|{'query_order':order})
  ms={n:{'median_us':statistics.median(b['median_us'] for b in bs),'blocks':bs} for n,bs in blocks.items()}
  r['trials'].append({'trial':trial,'fresh_key':True,'parameters':[50,50,56],'security':'SEAL tc128 accepted','response_bytes':len(expected[0]),'exact_server_byte_checks':len(qs),'exact_decryption_checks':len(qs),'max_error':max(checks),'invalid_queries_rejected':4,'output_ownership_checked':True,'measurements':ms,'method_orders':orders})
  r['summary_us']={n:statistics.median(t['measurements'][n]['median_us'] for t in r['trials']) for n in methods};r['fastest_plaintext']=min((n for n in methods if n.startswith('plaintext')),key=lambda n:r['summary_us'][n]);r['overhead']=r['summary_us']['pipeline_packed']/r['summary_us'][r['fastest_plaintext']];r['all_passed']=True
  (ROOT/args.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'trial':trial,'summary':r['summary_us'],'overhead':r['overhead']},indent=2),flush=True)
  if args.gpu:pg.close();cpu.close()
  c.close();g.close()
if __name__=='__main__':main()
