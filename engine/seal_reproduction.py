# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Separate Microsoft SEAL implementation of the ring-identity experiment.

Does not import the OpenFHE implementation, planner, state class, or helpers.
This is an implementation cross-check by the same author, not external review.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json
import time
import statistics
from pathlib import Path
import numpy as np
import seal

S=2**46
ROOT=Path(__file__).resolve().parent

def time_call(fn,count=9):
    fn();samples=[]
    for _ in range(count):
        start=time.perf_counter_ns();out=fn();samples.append((time.perf_counter_ns()-start)*1e-9)
    return out,{'median_s':statistics.median(samples),'samples_s':samples}

class Server:
    def __init__(self,context,encrypted):
        self.context=context;self.encrypted=tuple(encrypted)
        self.encoder=seal.CKKSEncoder(context);self.eval=seal.Evaluator(context)
        self.norm=self.eval.add_many([self.eval.square(c) for c in encrypted])

    @staticmethod
    def k(q):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(32,) or not np.all(np.isfinite(q)) or np.any(np.abs(q)>.5):raise ValueError('Bad query')
        return np.rint(-2*S*q).astype(np.int64)

    def shifted(self,state,oldk,newk):
        out=seal.Ciphertext(state)
        for j in np.flatnonzero(oldk!=newk):
            delta=int(newk[j])-int(oldk[j])
            p=self.encoder.encode(delta/S,float(S))
            self.eval.add_inplace(out,self.eval.multiply_plain(self.encrypted[int(j)],p))
        return out

    def fresh(self,q):return self.shifted(self.norm,np.zeros(32,dtype=np.int64),self.k(q))

    def ordinary(self,q):
        out=seal.Ciphertext(self.norm)
        for c,v in zip(self.encrypted,q):
            if v:self.eval.add_inplace(out,self.eval.multiply_plain(c,self.encoder.encode(float(-2*v),float(S))))
        return out

    def affine(self,a,b,c,qa,qb,qc,q):
        if not np.array_equal(self.k(q),self.k(qa)+self.k(qb)-self.k(qc)):raise ValueError('Certificate rejected')
        return self.eval.sub(self.eval.add(a,b),c)

    def reply(self,core,q,compress):
        out=self.eval.add_plain(core,self.encoder.encode(float(q@q),float(S*S)))
        return self.eval.rescale_to_next(out) if compress else out

def setup(x):
    p=seal.EncryptionParameters(seal.scheme_type.ckks);p.set_poly_modulus_degree(8192)
    p.set_coeff_modulus(seal.CoeffModulus.Create(8192,[56,46,56]))
    context=seal.SEALContext(p,True,seal.sec_level_type.tc128)
    assert context.parameters_set(),context.parameter_error_message()
    keys=seal.KeyGenerator(context);encoder=seal.CKKSEncoder(context)
    encryptor=seal.Encryptor(context,keys.create_public_key())
    decryptor=seal.Decryptor(context,keys.secret_key())
    start=time.perf_counter_ns()
    encrypted=[encryptor.encrypt(encoder.encode(np.ascontiguousarray(v),float(S))) for v in x.T]
    enc_time=(time.perf_counter_ns()-start)*1e-9
    start=time.perf_counter_ns();server=Server(context,encrypted);prep_time=(time.perf_counter_ns()-start)*1e-9
    decode=lambda ct:np.array(encoder.decode(decryptor.decrypt(ct)))[:len(x)]
    return server,decode,{'input_encryption_s':enc_time,'norm_preparation_s':prep_time,
                         'data_modulus_primes':encrypted[0].coeff_modulus_size(),
                         'data_modulus_bits':context.first_context_data().total_coeff_modulus_bit_count(),
                         'key_modulus_bits':context.key_context_data().total_coeff_modulus_bit_count()}

def binary(ct):return ct.to_string(seal.compr_mode_type.none)

def main():
    rng=np.random.default_rng(9613);x=rng.uniform(-1,1,(4096,32));server,decode,prep=setup(x)
    checks=[]
    for trial in range(9):
        qa,qb,qc=[rng.integers(-100000,100000,32)/2**20 for _ in range(3)];q=qa+qb-qc
        a,b,c=[server.fresh(v) for v in (qa,qb,qc)]
        affine=lambda:server.affine(a,b,c,qa,qb,qc,q)
        same=binary(affine())==binary(server.fresh(q));assert same
        row={'trial':trial,'affine_byte_equality':same,'methods':{}}
        for compressed in (False,True):
            for name,core in [('full',lambda:server.ordinary(q)),('affine',affine)]:
                ct,t=time_call(lambda:server.reply(core(),q,compressed),11)
                y,dt=time_call(lambda:decode(ct),5);err=float(np.max(np.abs(y-np.sum((x-q)**2,axis=1))))
                assert err<1e-5,(name,compressed,err)
                row['methods'][name+('_compressed' if compressed else '_raw')]={
                    'server':t,'decode':dt,'phase_sum_s':t['median_s']+dt['median_s'],
                    'max_abs_error':err,'bytes':len(binary(ct))}
        checks.append(row)
    chains=[]
    for family,seed in [('random',723),('corner',724),('zero',725)]:
        rng=np.random.default_rng(seed)
        if family=='random':xx=rng.uniform(-1,1,(4096,32))
        elif family=='corner':xx=np.tile(np.where(np.arange(4096)%2,1.,-1.)[:,None],(1,32))
        else:xx=np.zeros((4096,32))
        srv,dec,_=setup(xx);q=rng.uniform(-.5,.5,32);k=srv.k(q);state=srv.fresh(q);points=[]
        for i in range(10000):
            q=q.copy();q[i%32]=rng.uniform(-.5,.5);newk=srv.k(q);state=srv.shifted(state,k,newk);k=newk
            if i in (0,7,8,127,999,4999,9999):
                equal=binary(state)==binary(srv.fresh(q));err=float(np.max(np.abs(dec(srv.reply(state,q,True))-np.sum((xx-q)**2,axis=1))))
                assert equal and err<1e-5,(family,i,equal,err)
                points.append({'step':i+1,'byte_identical_to_fresh':equal,'max_abs_error':err})
        chains.append({'family':family,'seed':seed,'steps':10000,'fresh_key':True,'checkpoints':points})
        print('SEAL CHAIN',family,points[-1],flush=True)
    summary={name:{'server_ms':statistics.median(r['methods'][name]['server']['median_s'] for r in checks)*1000,
                   'phase_ms':statistics.median(r['methods'][name]['phase_sum_s'] for r in checks)*1000}
             for name in checks[0]['methods']}
    summary['raw_server_speedup']=summary['full_raw']['server_ms']/summary['affine_raw']['server_ms']
    summary['compressed_local_speedup']=summary['full_compressed']['phase_ms']/summary['affine_compressed']['phase_ms']
    data={'implementation':'Microsoft SEAL through seal-python 4.4.0','external_independent_reproduction':False,
          'security':'SEAL tc128','ring_dimension':8192,'scale_bits':46,'coefficient_modulus_bits':[56,46,56],
          'evaluation_keys':False,'threads':1,'records':4096,'features':32,'preparation':prep,
          'summary':summary,'trials':checks,'chains':chains}
    (ROOT/'results'/'seal_reproduction.json').write_text(json.dumps(data,indent=2)+'\n')
    print('SEAL SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':main()
