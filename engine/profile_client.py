# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Attribute client cost without moving any secret-key work to the server."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json, time, statistics
from pathlib import Path
import numpy as np
import seal
from seal_reproduction import Server, S, binary
from raw_backend import RawDirectionServer

ROOT = Path(__file__).resolve().parent

def fixture(seed=72166,bits=(56,46,56),records=None):
    rng=np.random.default_rng(seed);x=rng.uniform(-1,1,(4096,32)) if records is None else np.asarray(records,dtype=np.float64)
    assert x.shape==(4096,32) and np.all(np.isfinite(x)) and np.max(np.abs(x))<=1
    p=seal.EncryptionParameters(seal.scheme_type.ckks);p.set_poly_modulus_degree(8192)
    p.set_coeff_modulus(seal.CoeffModulus.Create(8192,list(bits)))
    context=seal.SEALContext(p,True,seal.sec_level_type.tc128)
    assert context.parameters_set()
    keys=seal.KeyGenerator(context);encoder=seal.CKKSEncoder(context)
    encryptor=seal.Encryptor(context,keys.create_public_key());decryptor=seal.Decryptor(context,keys.secret_key())
    encrypted=[encryptor.encrypt(encoder.encode(np.ascontiguousarray(v),float(S))) for v in x.T]
    server=Server(context,encrypted)
    return {'rng':rng,'x':x,'params':p,'context':context,'keys':keys,'encoder':encoder,'decryptor':decryptor,'server':server}

def query_case(f):
    rng=f['rng'];server=f['server']
    a,b,c=[rng.integers(-100000,100000,32)/2**20 for _ in range(3)];q=a+b-c
    ca,cb,cc=[server.fresh(v) for v in (a,b,c)];delta=server.eval.sub(cb,cc)
    raw=RawDirectionServer(f['context'],ca,delta,server.k(a),server.k(b)-server.k(c))
    blob=raw.reply(q);ct=seal.Ciphertext();ct.load_bytes(f['context'],blob)
    return q,ct,blob,raw

def measure(fn, count=101):
    for _ in range(5):fn()
    samples=[]
    for _ in range(count):
        start=time.perf_counter_ns();out=fn();samples.append((time.perf_counter_ns()-start)*1e-9)
    return {'median_us':statistics.median(samples)*1e6,'samples_s':samples}

def main():
    f=fixture();q,ct,blob,raw=query_case(f)
    context=f['context'];decoder=f['decryptor'];encoder=f['encoder']
    rows=[]
    for compressed in (False,True):
        target=f['server'].eval.rescale_to_next(ct) if compressed else ct
        data=binary(target);plain=decoder.decrypt(target);values=encoder.decode(plain)
        reuse_ct=seal.Ciphertext();reuse_plain=seal.Plaintext()
        def load():
            c=seal.Ciphertext();c.load_bytes(context,data);return c
        def client_old():return np.array(encoder.decode(decoder.decrypt(load())))
        def client_nocopy():return encoder.decode(decoder.decrypt(load()))
        def client_reuse():
            reuse_ct.load_bytes(context,data);decoder.decrypt(reuse_ct,reuse_plain);return encoder.decode(reuse_plain)
        funcs={'load_new':load,'load_reused':lambda:reuse_ct.load_bytes(context,data),
               'decrypt_new':lambda:decoder.decrypt(target),'decrypt_reused':lambda:decoder.decrypt(target,reuse_plain),
               'decode_to_numpy':lambda:encoder.decode(plain),'extra_numpy_copy':lambda:np.array(values),
               'client_old':client_old,'client_nocopy':client_nocopy,'client_reused':client_reuse}
        result={name:measure(fn) for name,fn in funcs.items()}
        err=float(np.max(np.abs(client_old()-np.sum((f['x']-q)**2,axis=1))))
        assert err<1e-5
        row={'compressed':compressed,'bytes':len(data),'components':target.size(),'limbs':target.coeff_modulus_size(),'error':err,'phases':result}
        rows.append(row);print(compressed,{k:round(v['median_us'],3) for k,v in result.items()},flush=True)
    (ROOT/'results'/'profile.json').write_text(json.dumps({'cases':rows},indent=2)+'\n')

if __name__=='__main__':main()
