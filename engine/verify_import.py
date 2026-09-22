# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Differential import rejection, exact residues, workload corners and fallback."""
import ctypes,json,struct
from pathlib import Path
import numpy as np
import seal
from profile_client import fixture
from native_client import NativeClient
from benchmark_import import configure,verify,stats,CONFIGS
from seal_reproduction import binary
from reproduce import machine
ROOT=Path(__file__).resolve().parent
def main():
 if not __debug__:raise RuntimeError('Assertions required')
 rng=np.random.default_rng(240818);rows=[];corners=[];fallback=[]
 for label,bits in CONFIGS.items():
  f=fixture(98133,bits);s=f['server'];q=np.zeros(32);blob=binary(s.reply(s.fresh(q),q,False));c=NativeClient(f['params'],f['keys'].secret_key(),True);old=NativeClient(f['params'],f['keys'].secret_key(),True,'client_retained.so');configure(c)
  def outcome(cl,b,m):
   try:return True,cl.decode(b,m)
   except ValueError:return False,None
  accepted=rejected=0
  def check(b):
   nonlocal accepted,rejected
   ok,ref=outcome(old,b,8);got,y=outcome(c,b,10);assert got==ok,'Acceptance mismatch'
   if ok:
    verify(c,b);assert np.allclose(y,ref,rtol=1e-12,atol=1e-9,equal_nan=True);accepted+=1
   else:rejected+=1
  # Every bit of the complete serialized metadata. Acceptance follows SEAL.
  for i in range(113):
   for bit in range(8):
    b=bytearray(blob);b[i]^=1<<bit;check(bytes(b))
  for level in [None,'rescaled']:
   if level:
    ct=seal.Ciphertext();ct.load_bytes(f['context'],blob);ct=s.eval.rescale_to_next(ct);base=binary(ct);mod=f['context'].get_context_data(ct.parms_id()).parms().coeff_modulus()
   else:base=blob;mod=f['context'].first_context_data().parms().coeff_modulus()
   check(base)
   for component in range(3):
    for limb,p in enumerate(mod):
     for index in [0,1,4095,4096,8190,8191]:
      for v in [0,p.value()-1,p.value(),2**64-1]:
       b=bytearray(base);struct.pack_into('<Q',b,113+8*((component*len(mod)+limb)*8192+index),v);check(bytes(b))
   for _ in range(128):
    b=bytearray(base);i=int(rng.integers(113,len(b)));b[i]^=1<<int(rng.integers(8));check(bytes(b))
   for n in [0,1,15,16,112,113,len(base)-1,len(base)-8]:check(base[:n])
   check(base+b'x');check(base+b'\0'*16)
   for value in [float('nan'),float('inf'),-1.,0.,2.**90,2.**94,2.**200]:
    b=bytearray(base);struct.pack_into('<d',b,73,value);check(bytes(b))
  check(binary(s.encrypted[0]));check(b'\0'*(1024*1024+1));verify(c,blob)
  assert np.max(np.abs(c.decode(blob,10)-np.sum(f['x']**2,axis=1)))<1e-5
  st=stats(c);assert st['fast']>0 and st['fallback']>0
  rows.append({'config':label,'accepted_and_exactly_decrypted':accepted,'rejected':rejected,'import_stats':st,'recovery_after_invalid_input':True});c.close();old.close()
 # Fresh keys, bounded adversarial slot layouts, raw and rescaled answers.
 for kind in ['all_positive','alternating_slots','random_signs','zero']:
  if kind=='all_positive':x=np.ones((4096,32))
  elif kind=='alternating_slots':x=np.tile(np.where(np.arange(4096)%2,1.,-1.)[:,None],(1,32))
  elif kind=='random_signs':x=rng.choice([-1.,1.],(4096,32))
  else:x=np.zeros((4096,32))
  for label,bits in CONFIGS.items():
   f=fixture(80013,bits,x);s=f['server'];c=NativeClient(f['params'],f['keys'].secret_key(),True);configure(c)
   qs=[np.full(32,-.5),np.full(32,.5),np.zeros(32),rng.choice([-.5,.5],32),rng.uniform(-.5,.5,32)]
   for q in qs:
    ct=s.reply(s.fresh(q),q,False);target=np.sum((x-q)**2,axis=1)
    for level in [ct,s.eval.rescale_to_next(ct)]:
     b=binary(level);verify(c,b);y=c.decode(b,10);ref=f['encoder'].decode(f['decryptor'].decrypt(level));err=float(np.max(np.abs(y-target)));diff=float(np.max(np.abs(y-ref)));assert err<1e-5 and diff<1e-9,(label,kind,err,diff)
     corners.append({'config':label,'kind':kind,'limbs':level.coeff_modulus_size(),'max_target':float(np.max(target)),'error':err,'seal_difference':diff})
   c.close()
 for bits in [(60,30,56),(46,56,56)]:
  p=seal.EncryptionParameters(seal.scheme_type.ckks);p.set_poly_modulus_degree(8192);p.set_coeff_modulus(seal.CoeffModulus.Create(8192,list(bits)));ctx=seal.SEALContext(p,True,seal.sec_level_type.tc128);assert ctx.parameters_set();k=seal.KeyGenerator(ctx);e=seal.Encryptor(ctx,k.create_public_key());enc=seal.CKKSEncoder(ctx);ev=seal.Evaluator(ctx);c=NativeClient(p,k.secret_key(),True);old=NativeClient(p,k.secret_key(),True,'client_retained.so');configure(c)
  for _ in range(4):
   a=rng.uniform(-.25,.25,4096);b=rng.uniform(-.25,.25,4096);ct=ev.multiply(e.encrypt(enc.encode(a,2.**30)),e.encrypt(enc.encode(b,2.**30)));blob=binary(ct);verify(c,blob);y=c.decode(blob,10);ref=old.decode(blob,8);err=float(np.max(np.abs(y-a*b)));assert err<1e-5 and np.max(np.abs(y-ref))<1e-9;fallback.append({'bits':bits,'error':err})
  assert stats(c)['fast']==0 and stats(c)['fallback']>0;c.close();old.close()
 r={'machine':machine(),'all_passed':True,'import_differential':rows,'corner_checks':corners,'fallback_checks':fallback}
 (ROOT/'results/import_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'all_passed':True,'import_differential':rows,'corner_count':len(corners),'corner_max_error':max(v['error'] for v in corners),'fallback_count':len(fallback)},indent=2))
if __name__=='__main__':main()
