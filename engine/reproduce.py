# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Portable Linux x86-64 validation and rotating-query timing, no uploaded keys."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,ctypes,hashlib,json,platform,shutil,statistics,struct,subprocess,time
from pathlib import Path
import numpy as np
import seal
from profile_client import fixture,measure
from native_client import NativeClient,NativeServer
from raw_backend import RawDirectionServer
from seal_reproduction import binary,S
ROOT=Path(__file__).resolve().parent

def machine():
    cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    return {'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,
        'cpu_model':next((line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')),platform.processor()),
        'cpu_flags':next((line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('flags')),''),
        'nvidia_smi':shutil.which('nvidia-smi'),'gpu_device_paths':[p for p in ('/dev/dri','/dev/nvidia0','/dev/kfd') if Path(p).exists()],
        'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in ('.py','.cpp','.c','.cu')}}

def validate(f,hexl,blobs,targets,miss_blobs,miss_targets):
    client=NativeClient(f['params'],f['keys'].secret_key(),hexl)
    checks=[];vector_checks=[]
    for label,sequence,expected in [('queries',blobs,targets),('changed_tail',miss_blobs,miss_targets)]:
        for blob,target in zip(sequence,expected):
            ct=seal.Ciphertext();ct.load_bytes(f['context'],blob)
            for level in (ct,f['server'].eval.rescale_to_next(ct)):
                data=binary(level);ref=f['encoder'].decode(f['decryptor'].decrypt(level))
                if hexl:
                    assert client.verify_vector_decryption(data),'Vector residue mismatch'
                    result=client.decode(data,6)
                    err=float(np.max(np.abs(result-target)));difference=float(np.max(np.abs(result-ref)))
                    assert err<1e-5 and difference<1e-9
                    vector_checks.append({'error':err,'reference_difference':difference,'limbs':level.coeff_modulus_size()})
                # First access may miss; second must hit. Compare exact modular decryption.
                before=client.tail_stats()
                for repeat in range(2):
                    assert client.verify_cached_decryption(data),'Exact residue mismatch'
                    result=client.decode(data,5)
                    err=float(np.max(np.abs(result-target)));difference=float(np.max(np.abs(result-ref)))
                    assert err<1e-5 and difference<1e-9,(label,err,difference)
                    checks.append({'kind':label,'limbs':level.coeff_modulus_size(),'repeat':repeat,'error':err,'reference_difference':difference})
                assert client.tail_stats()['hits']>=before['hits']+3
    # Explicitly verify that different heads with identical tails take the reuse branch.
    hit_client=NativeClient(f['params'],f['keys'].secret_key(),hexl)
    for blob in blobs:hit_client.decode(blob,5)
    assert hit_client.tail_stats()=={'hits':len(blobs)-1,'misses':1}
    for blob in miss_blobs:hit_client.decode(blob,5)
    assert hit_client.tail_stats()['misses']==1+len(miss_blobs)
    # Fully valid changed tail must recompute, even if just its final coefficient changes.
    mutated=bytearray(blobs[0]);p=f['context'].first_context_data().parms().coeff_modulus()[1].value()
    last=struct.unpack_from('<Q',mutated,len(mutated)-8)[0]
    struct.pack_into('<Q',mutated,len(mutated)-8,(last+1)%p)
    before=hit_client.tail_stats();assert hit_client.verify_cached_decryption(bytes(mutated))
    assert hit_client.tail_stats()['misses']==before['misses']+1
    # Invalid import must fail before any cached arithmetic executes.
    blob=blobs[0];scale=bytearray(blob);struct.pack_into('<d',scale,73,float('nan'))
    tail=bytearray(blob);struct.pack_into('<Q',tail,len(tail)-8,2**64-1)
    malformed=[blob[:12],blob[:-8],blob+b'x',bytes(scale),bytes(tail),binary(f['server'].encrypted[0])]
    before=hit_client.tail_stats()
    for bad in malformed:
        try:hit_client.decode(bad,5)
        except ValueError:pass
        else:raise AssertionError('Invalid ciphertext accepted')
    assert hit_client.tail_stats()==before
    if hexl:
        for bad in malformed:
            try:client.decode(bad,6)
            except ValueError:pass
            else:raise AssertionError('Vector path accepted invalid ciphertext')
    result={'numerical_checks':checks,'vector_checks':vector_checks,'exact_residue_checks':len(checks)+1+len(vector_checks),'invalid_replies_rejected':len(malformed)*(2 if hexl else 1),
        'cache_hit_and_miss_assertions':True,'key_scoped_first_access_missed':True,'all_passed':True}
    client.close();hit_client.close();return result

def rotating(functions,order):
    state=[0]
    def call():
        index=order[state[0]%len(order)];state[0]+=1
        return functions[index]()
    return call

def main():
    if not __debug__:raise RuntimeError('Run without Python -O: validation assertions must remain enabled')
    ap=argparse.ArgumentParser();ap.add_argument('--hexl',action='store_true');ap.add_argument('--trials',type=int,default=5)
    ap.add_argument('--queries',type=int,default=8);ap.add_argument('--samples',type=int,default=64);ap.add_argument('--blocks',type=int,default=3)
    ap.add_argument('--seed',type=int,default=120301);ap.add_argument('--output',type=Path,default=ROOT/'results'/'reproduction.json')
    args=ap.parse_args()
    if min(args.trials,args.queries,args.samples,args.blocks)<1:ap.error('Counts must be positive')
    if args.queries<2:ap.error('At least two distinct queries are required to exercise cache misses')
    record={'machine':machine(),'configuration':vars(args)|{'output':str(args.output)},'trials':[],
        'scope':'Warm local public-query/encrypted-database distance workload; rotating prepared queries, full serialized replies and owned 4096-double outputs. Setup and network excluded from warm timings; setup recorded separately. No external replication is claimed.',
        'parameters':{'poly_modulus_degree':8192,'coeff_modulus_bits':[56,46,56],'input_scale_bits':46,'security':'SEAL tc128'},
        'independent_reproduction':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for trial in range(args.trials):
        start=time.perf_counter_ns();f=fixture(args.seed+trial);fixture_s=(time.perf_counter_ns()-start)*1e-9
        s=f['server'];rng=f['rng'];base=s.fresh(np.zeros(32));norm=np.ascontiguousarray(np.sum(f['x']**2,axis=1))
        native_servers=[];blobs=[];queries=[];targets=[];plain_directions=[];ks=[]
        start=time.perf_counter_ns()
        for _ in range(args.queries):
            q=np.ascontiguousarray(rng.integers(-100000,100000,32)/2**20);delta=s.eval.sub(s.fresh(q),base)
            raw=RawDirectionServer(f['context'],base,delta,s.k(np.zeros(32)),s.k(q))
            server=NativeServer(f['params'],raw,args.hexl);blob=server.reply(q)
            assert blob==binary(s.reply(s.fresh(q),q,False))
            native_servers.append(server);blobs.append(blob);queries.append(q);targets.append(np.sum((f['x']-q)**2,axis=1))
            plain_directions.append(np.ascontiguousarray(-2*f['x']@q));ks.append(np.ascontiguousarray(s.k(q)))
        preparation_s=(time.perf_counter_ns()-start)*1e-9
        assert len(set(blobs))==len(blobs),'Queries must produce distinct ciphertexts'
        assert len({blob[113+2*2*8192*8:] for blob in blobs})==1,'Expected invariant component'
        miss_blobs=[];miss_targets=[];encryptor=seal.Encryptor(f['context'],f['keys'].create_public_key())
        for _ in range(args.queries):
            a=rng.uniform(-1,1,4096);b=rng.uniform(-1,1,4096)
            aa=encryptor.encrypt(f['encoder'].encode(a,float(S)));bb=encryptor.encrypt(f['encoder'].encode(b,float(S)))
            miss_blobs.append(binary(s.eval.multiply(aa,bb)));miss_targets.append(a*b)
        validation=validate(f,args.hexl,blobs,targets,miss_blobs,miss_targets)
        start=time.perf_counter_ns();client=NativeClient(f['params'],f['keys'].secret_key(),args.hexl);client_setup_s=(time.perf_counter_ns()-start)*1e-9
        client.lib.plain_pipeline.argtypes=[ctypes.c_void_p]*5;client.lib.plain_pipeline.restype=ctypes.c_int
        def plain(i):
            out=np.empty(4096)
            assert client.lib.plain_pipeline(queries[i].ctypes.data,ks[i].ctypes.data,norm.ctypes.data,plain_directions[i].ctypes.data,out.ctypes.data)
            return out
        for i in range(args.queries):assert np.max(np.abs(plain(i)-targets[i]))<1e-10
        def python_decode(blob):
            ct=seal.Ciphertext();ct.load_bytes(f['context'],blob)
            return np.array(f['encoder'].decode(f['decryptor'].decrypt(ct)))
        methods={
          'client_baseline':[lambda b=b:client.decode(b,4) for b in blobs],
          'client_tail_cache':[lambda b=b:client.decode(b,5) for b in blobs],
          'client_all_miss_baseline':[lambda b=b:client.decode(b,4) for b in miss_blobs],
          'client_all_miss_tail_cache':[lambda b=b:client.decode(b,5) for b in miss_blobs],
          'pipeline_baseline':[lambda i=i:client.decode(native_servers[i].reply(queries[i]),4) for i in range(args.queries)],
          'pipeline_tail_cache':[lambda i=i:client.decode(native_servers[i].reply(queries[i]),5) for i in range(args.queries)],
          'pipeline_python_client':[lambda i=i:python_decode(native_servers[i].reply(queries[i])) for i in range(args.queries)],
          'plaintext_pipeline':[lambda i=i:plain(i) for i in range(args.queries)]}
        if args.hexl:
            methods.update({'client_vector':[lambda b=b:client.decode(b,6) for b in blobs],
                'pipeline_vector':[lambda i=i:client.decode(native_servers[i].reply(queries[i]),6) for i in range(args.queries)]})
        blocks={name:[] for name in methods}
        for block in range(args.blocks):
            order=rng.permutation(args.queries).tolist();names=list(methods);rng.shuffle(names)
            for name in names:
                # Prime the current schedule separately, then time one complete rotating sweep.
                fn=rotating(methods[name],order)
                for _ in range(args.queries):fn()
                blocks[name].append(measure(fn,args.samples)|{'query_order':order})
        measurements={name:{'median_us':statistics.median(b['median_us'] for b in bs),'blocks':bs} for name,bs in blocks.items()}
        row={'trial':trial,'seed':args.seed+trial,'fresh_random_key':True,'fixture_setup_s':fixture_s,'prepared_query_setup_s':preparation_s,
          'client_setup_s':client_setup_s,'reply_bytes':len(blobs[0]),'extra_cache_bytes':2*2*8192*8,
          'validation':validation,'measurements':measurements}
        record['trials'].append(row)
        record['summary_us']={name:statistics.median(t['measurements'][name]['median_us'] for t in record['trials']) for name in methods}
        su=record['summary_us'];record['ratios']={'client_cache_speedup':su['client_baseline']/su['client_tail_cache'],
          'pipeline_cache_speedup':su['pipeline_baseline']/su['pipeline_tail_cache'],
          'all_miss_slowdown':su['client_all_miss_tail_cache']/su['client_all_miss_baseline'],
          'cached_pipeline_vs_plaintext':su['pipeline_tail_cache']/su['plaintext_pipeline']}
        if args.hexl:record['ratios'].update({'client_vector_speedup':su['client_baseline']/su['client_vector'],
          'pipeline_vector_speedup':su['pipeline_baseline']/su['pipeline_vector'],
          'vector_pipeline_vs_plaintext':su['pipeline_vector']/su['plaintext_pipeline']})
        record['all_validation_passed']=all(t['validation']['all_passed'] for t in record['trials'])
        args.output.write_text(json.dumps(record,indent=2)+'\n')
        print(trial,{name:round(value['median_us'],2) for name,value in measurements.items()},flush=True)
        client.close()
        for server in native_servers:server.close()
    print(json.dumps(record['ratios'],indent=2),flush=True)
if __name__=='__main__':main()
