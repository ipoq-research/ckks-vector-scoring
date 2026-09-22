# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Reproducible CKKS scoring research experiment. See README.md for scope and licenses."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, ctypes, hashlib, json, math, secrets, shutil, statistics, struct, subprocess, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from verify_receipt import canonical, data_values, data_digest, ROWS, COLS, TOLERANCE, query_for_seed
ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seed_value(text):
    n = int(text)
    require(0 <= n < 2**64, 'Seed must be between 0 and 2^64 - 1')
    return n


def validate_query(q):
    require(isinstance(q, list) and len(q) == COLS, 'Query must be a JSON array of 32 coordinates')
    require(all(type(v) in (float, int) and math.isfinite(v) and abs(v) <= .5 for v in q), 'Every coordinate must be finite and within [-0.5, 0.5]')
    return [float(v) for v in q]


def cpu_model():
    path = Path('/proc/cpuinfo')
    if path.exists():
        return next((line.split(':', 1)[1].strip() for line in path.read_text().splitlines() if line.startswith('model name')), 'Unknown CPU')
    return 'Unknown CPU'


def preflight(engine, backend):
    needed = ['client_native_hexl.so', 'fresh_cpu.so'] + (['fresh_cuda.so'] if backend == 'gpu' else [])
    require(all((engine/n).is_file() for n in needed), 'Native libraries missing. Run scripts/build.sh for the selected backend first.')
    info = {'cpu': cpu_model(), 'backend': backend, 'cpu_threads_configured': 1, 'gpu': None}
    if backend == 'gpu':
        flags = Path('/proc/cpuinfo').read_text()
        require('avx512ifma' in flags and 'avx512dq' in flags, 'This GPU experiment requires a CPU with AVX-512 IFMA/DQ. No silent fallback.')
        info['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True, timeout=10).strip()
    h = hashlib.sha256()
    for n in needed:
        h.update(n.encode());h.update(hashlib.sha256((engine/n).read_bytes()).digest())
    return info, h.hexdigest()


class Session:
    def __init__(self, args):
        self.args = args
        self.hardware, self.binary_commitment = preflight(args.engine, args.backend)
        sys.path.insert(0, str(args.engine))
        import numpy as np
        import seal
        from profile_client import fixture
        from packed_backend import PackedBackend
        from fresh_backend import PlainGPU
        from native_client import NativeClient
        self.np, self.seal = np, seal
        self.session_id = str(uuid.uuid4()); self.index = 0
        values = data_values(args.seed)
        self.x = np.asarray(values, dtype=np.float64).reshape(ROWS, COLS)
        self.dataset_digest = data_digest(values)
        print('\nIPOQ | ENCRYPTED SCORING RESEARCH', flush=True)
        print('Mode: ' + ('GPU live benchmark' if args.backend == 'gpu' else 'CPU reproduction; does not reproduce the GPU headline'), flush=True)
        print('Dataset seed:', args.seed, '\nDataset commitment:', self.dataset_digest, flush=True)
        print('Session:', self.session_id, '\nBuild commitment:', self.binary_commitment, flush=True)
        started = time.perf_counter()
        self.f = fixture(args.seed, (50, 50, 56), self.x)
        self.g = PackedBackend(self.f, args.backend == 'gpu')
        self.cpu = PackedBackend(self.f, False) if args.backend == 'gpu' else self.g
        self.client = NativeClient(self.f['params'], self.f['keys'].secret_key(), True)
        lib = self.client.lib
        lib.client_verify_fast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.client_verify_fast.restype = ctypes.c_int
        self.xx = np.ascontiguousarray(self.x.T)
        self.norm = np.ascontiguousarray(np.sum(self.x*self.x, axis=1))
        self.pg = PlainGPU(self.g.lib, self.xx, self.norm) if args.backend == 'gpu' else None
        self.setup_s = time.perf_counter() - started
        digest = hashlib.sha256()
        for ct in self.f['server'].encrypted:
            digest.update(ct.to_string(seal.compr_mode_type.none))
        self.encrypted_commitment = digest.hexdigest()
        print('Encrypted database commitment:', self.encrypted_commitment, flush=True)
        print('Fresh keys and encrypted records prepared in %.3f s (setup excluded from online timing).' % self.setup_s, flush=True)
        print('Dataset and encrypted database are now fixed. Choose the challenge query.', flush=True)

    def close(self):
        if self.pg is not None:self.pg.close()
        if self.cpu is not self.g:self.cpu.close()
        self.g.close();self.client.close()

    def plain_cpu(self, q):
        np=self.np;out=np.empty(ROWS)
        require(self.cpu.lib.fresh_plain(self.xx.ctypes.data, self.norm.ctypes.data, q.ctypes.data, out.ctypes.data), 'Plaintext control failed')
        return out

    def plain_blas(self, q):
        return self.norm - 2*(q @ self.xx) + q @ q

    def run(self, q, query_seed):
        dest = None
        for event in self.run_steps(q, query_seed):
            if event["phase"] == "exported": dest = Path(event["receipt_directory"])
        return dest

    def run_steps(self, q, query_seed):
        np, seal = self.np, self.seal
        self.index += 1
        qs = [np.ascontiguousarray(validate_query(q), dtype=np.float64)]
        qs += [np.ascontiguousarray(query_for_seed(query_seed, i), dtype=np.float64) for i in range(1, 16)]
        print('\nChallenge %d | selected query + 15 rotating companion queries' % self.index, flush=True)
        print('Query seed:', query_seed, '\nSelected coordinates:', json.dumps(qs[0].tolist()), flush=True)
        outputs=[];saved=[];byte_checks=0;ref_differences=[];errors=[]
        for i, query in enumerate(qs):
            blob=self.g.reply_packed(query);saved.append(blob)
            result=self.client.decode(blob,10)
            require(result.shape == (ROWS,) and np.all(np.isfinite(result)), 'Invalid score vector')
            expected=np.sum((self.x-query)**2,axis=1)
            error=float(np.max(np.abs(result-expected)))
            require(error <= TOLERANCE, 'Output outside agreed tolerance')
            require(self.client.lib.client_verify_fast(self.client.handle,blob,len(blob)), 'Exact decryption reference check failed')
            ct=seal.Ciphertext();ct.load_bytes(self.f['context'],blob)
            reference=self.f['encoder'].decode(self.f['decryptor'].decrypt(ct))
            difference=float(np.max(np.abs(result-reference)))
            require(difference <= 1e-9, 'SEAL client comparison failed')
            srv=self.f['server'];ref_ct=srv.reply(srv.fresh(query),query,False)
            ref_output=self.f['encoder'].decode(self.f['decryptor'].decrypt(ref_ct))
            require(np.max(np.abs(result-ref_output)) <= 1e-8, 'SEAL server score comparison failed')
            # Dyadic grid queries allow exact reference ciphertext equality. Arbitrary
            # decimal queries retain their values and use the numerical comparison.
            if np.all(query*2**20 == np.rint(query*2**20)):
                require(blob == ref_ct.to_string(seal.compr_mode_type.none), 'Exact server reply comparison failed')
                byte_checks += 1
            for control in [self.plain_cpu,self.plain_blas] + ([self.pg.reply] if self.pg else []):
                require(np.max(np.abs(control(query)-expected)) <= 1e-10, 'Plaintext control mismatch')
            errors.append(error);ref_differences.append(difference)
            outputs.append({'index':i,'query':query.tolist(),'decrypted_scores':result.tolist(),
                            'ciphertext_sha256':hashlib.sha256(blob).hexdigest()})
        for blob, row in zip(saved, outputs):
            require(hashlib.sha256(blob).hexdigest() == row['ciphertext_sha256'], 'Returned reply mutated')
        print('PASS | 65,536 returned scores checked; maximum error %.3e' % max(errors), flush=True)
        print('PASS | %d exact encrypted-reply checks; 16 exact decryption checks' % byte_checks, flush=True)
        yield {"phase":"scored", "scores_checked":65536, "maximum_error":max(errors),
               "exact_ciphertext_checks":byte_checks, "selected_query":qs[0].tolist(), "query_seed":query_seed}
        b=saved[0]
        malformed=[b[:-8],b+b'invalid-trailer',b[:-8]+b'\xff'*8]
        rejected=0
        for bad in malformed:
            try:self.client.decode(bad,10)
            except ValueError:rejected+=1
            else:raise ValueError('Malformed ciphertext accepted')
        require(np.max(np.abs(self.client.decode(self.g.reply_packed(qs[0]),10)-np.asarray(outputs[0]['decrypted_scores']))) <= 1e-9, 'Recovery after malformed reply failed')
        print('PASS | %d malformed replies rejected; normal scoring recovered' % rejected, flush=True)
        print('This checks malformed serialization, not authentication of arbitrary valid ciphertext changes.', flush=True)
        yield {"phase":"verified", "malformed_replies_rejected":rejected, "recovery_passed":True}
        methods={'encrypted_pipeline':lambda query:self.client.decode(self.g.reply_packed(query),10),
                 'plaintext_native_cpu':self.plain_cpu,'plaintext_blas':self.plain_blas}
        if self.pg:methods['plaintext_gpu']=self.pg.reply
        rng=np.random.default_rng(secrets.randbits(64));measurements={n:[] for n in methods};orders=[]
        print('Timing full local requests and matched plaintext controls...', flush=True)
        for block in range(self.args.blocks):
            names=list(methods);rng.shuffle(names);orders.append(names)
            for name in names:
                fn=methods[name];order=rng.permutation(16).tolist()
                for j in range(5):fn(qs[order[j%16]])
                samples=[]
                for j in range(self.args.samples):
                    start=time.perf_counter_ns();answer=fn(qs[order[j%16]])
                    samples.append((time.perf_counter_ns()-start)/1000)
                # Numerical checks are outside the timed request.
                last=order[(self.args.samples-1)%16]
                require(np.max(np.abs(answer-np.sum((self.x-qs[last])**2,axis=1))) <= TOLERANCE, 'Timed method output failed validation')
                measurements[name].append({'query_order':order,'samples_us':samples,'median_us':statistics.median(samples)})
        medians={n:statistics.median(block['median_us'] for block in blocks) for n,blocks in measurements.items()}
        best=min((n for n in medians if n.startswith('plaintext_')),key=medians.get)
        overhead=medians['encrypted_pipeline']/medians[best]
        for name,value in medians.items():print('%-25s %10.3f us' % (name,value), flush=True)
        print('LIVE overhead: %.3fx versus %s' % (overhead,best), flush=True)
        print('10x median threshold: ' + ('MET on this run' if overhead <= 10 else 'NOT MET on this run'), flush=True)
        public={'schema':'ipoq-demo-receipt-v1','created_utc':datetime.now(timezone.utc).isoformat(),
                'session_id':self.session_id,'challenge':self.index,'dataset_seed':self.args.seed,'query_seed':query_seed,
                'shape':[ROWS,COLS],'dataset_generator':'SHAKE256-u32-v1','dataset_sha256':self.dataset_digest,
                'encrypted_database_sha256':self.encrypted_commitment,'native_build_sha256':self.binary_commitment,
                'hardware':self.hardware,'declared_tolerance':TOLERANCE,'setup_seconds_excluded':self.setup_s,
                'security_setting':'CKKS parameters accepted by SEAL tc128',
                'checks':{'scores_checked':65536,'maximum_observed_error':max(errors),'maximum_seal_client_difference':max(ref_differences),
                          'exact_ciphertext_reference_checks':byte_checks,'exact_decryption_reference_checks':16,
                          'numerical_seal_server_checks':16,'malformed_replies_rejected':rejected,'owned_reply_and_recovery_checks':True},
                'timing':{'keys_in_this_session':1,'queries':16,'blocks':self.args.blocks,'samples_per_method_per_block':self.args.samples,
                          'method_orders':orders,'methods':measurements,'medians_us':medians,'fastest_plaintext':best,
                          'overhead':overhead,'median_10x_met':overhead <= 10,'absolute_234_93_us_met':medians['encrypted_pipeline'] <= 234.93},
                'scope':'Live local synthetic-data scoring with public queries and a prepared encrypted database. Includes full reply construction, device transfers when applicable, validated import, decryption, full decoding and owned output. Network and setup excluded. Same-host roles, not isolation from the host operator. One fresh key per session; no independent timing/security attestation.',
                'queries':outputs}
        yield {"phase":"benchmarked", "medians_us":medians, "overhead":overhead, "median_10x_met":overhead <= 10}
        receipt={'payload':public,'sha256':hashlib.sha256(canonical(public)).hexdigest()}
        dest=self.args.output/self.session_id/('challenge_%02d' % self.index);dest.mkdir(parents=True,exist_ok=False)
        (dest/'receipt.json').write_bytes(canonical(receipt))
        shutil.copyfile(ROOT/'verify_receipt.py',dest/'verify_receipt.py')
        write_summary(public,dest/'receipt.md',receipt['sha256'])
        print('Shareable receipt:',dest,flush=True)
        yield {"phase":"exported", "receipt_directory":str(dest)}


def write_summary(p,path,digest):
    m=p['timing']['medians_us'];c=p['checks']
    lines=['# IPOQ live challenge receipt','', '**LIVE '+p['hardware']['backend'].upper()+' RUN** | '+p['created_utc'],'',
           'This receipt contains current measurements. The previously reported 9.85x GPU result is a separate archived experiment, not substituted here.','',
           '| Measurement | Result |','| --- | ---: |']
    for name,value in m.items():lines.append('| '+name.replace('_',' ')+' | %.3f µs |'%value)
    lines+=['| Overhead vs fastest tested plaintext | %.3fx |'%p['timing']['overhead'],'',
            '**10x median threshold:** '+('met' if p['timing']['median_10x_met'] else 'not met')+'.',
            '**Correctness:** %s scores checked; maximum observed error %.3e, against tolerance 1e-5.'%(c['scores_checked'],c['maximum_observed_error']),'',
            '%d exact ciphertext reference checks; 16 exact decryption checks; 16 SEAL server score comparisons; %d malformed replies rejected.'%(c['exact_ciphertext_reference_checks'],c['malformed_replies_rejected']),'',
            'Dataset seed: `%d`. Query seed: `%d`. Shape: 4,096 records × 32 features.'%(p['dataset_seed'],p['query_seed']),'',
            'Dataset commitment: `'+p['dataset_sha256']+'`','Encrypted database commitment: `'+p['encrypted_database_sha256']+'`',
            'Receipt SHA-256: `'+digest+'`','',
            '## Check every score independently','',
            'Run `python3 verify_receipt.py receipt.json`. This uses only the Python standard library to regenerate the public synthetic dataset and recompute every distance. It contains no FHE implementation.','',
            'The verifier checks numerical results and receipt consistency. A receipt hash is not a signature or an attestation of encrypted execution, security, performance, or author identity. Independent execution and cryptographic review remain necessary.','',
            '## Benchmark boundary','',p['scope'],'',
            'Malformed-reply rejection does not authenticate arbitrary well-formed changes or establish correct server evaluation. Trial medians are not a per-request latency guarantee.','',
            'Hardware: '+p['hardware']['cpu']+(' / '+p['hardware']['gpu'] if p['hardware']['gpu'] else '')+'.',
            'Each session uses one fresh key, 16 rotating queries, and %d blocks of %d samples per method.'%(p['timing']['blocks'],p['timing']['samples_per_method_per_block'])]
    path.write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine',type=Path,default=ROOT/'engine')
    ap.add_argument('--backend',choices=['cpu','gpu'],default='cpu')
    ap.add_argument('--seed',type=seed_value,default=None,help='Public synthetic dataset seed')
    ap.add_argument('--query-seed',type=seed_value,default=None)
    ap.add_argument('--query',help='Optional JSON array of 32 query coordinates')
    ap.add_argument('--interactive',action='store_true')
    ap.add_argument('--blocks',type=int,default=3)
    ap.add_argument('--samples',type=int,default=64)
    ap.add_argument('--output',type=Path,default=ROOT/'results')
    args=ap.parse_args()
    try:
        require(__debug__, 'Run without Python -O: native fixture validation must remain active')
        require(1 <= args.blocks <= 10 and 16 <= args.samples <= 2048, 'Use 1-10 blocks and 16-2048 samples')
        args.engine=args.engine.resolve();args.output=args.output.resolve()
        if args.seed is None:args.seed=secrets.randbits(64)
        session=Session(args)
        try:
            if not args.interactive:
                seed=args.query_seed if args.query_seed is not None else secrets.randbits(64)
                query=validate_query(json.loads(args.query)) if args.query else query_for_seed(seed)
                session.run(query,seed)
            else:
                while True:
                    line=input("\nEnter 'seed NUMBER', a 32-number JSON query, Enter for random, or 'quit': ").strip()
                    if line.lower() in ('quit','q','exit'):break
                    try:
                        seed=seed_value(line.split(maxsplit=1)[1]) if line.startswith('seed ') else secrets.randbits(64)
                        query=validate_query(json.loads(line)) if line.startswith('[') else query_for_seed(seed)
                        require(not line or line.startswith('seed ') or line.startswith('['),'Expected a seed command or JSON query')
                    except (ValueError,IndexError) as e:
                        print('Input rejected:',e);continue
                    session.run(query,seed)
        finally:session.close()
    except (ValueError,OSError,subprocess.SubprocessError,ImportError,EOFError,KeyboardInterrupt) as e:
        print('EXPERIMENT STOPPED: '+str(e),file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':sys.exit(main())
