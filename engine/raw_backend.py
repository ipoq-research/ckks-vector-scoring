# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Experimental native addition backend for a validated SEAL ciphertext layout.

Uses real encrypted residues. The unchanged nonlinear tail is shared internally
and copied into each complete reply. No secret key is accepted by this module.
"""
import ctypes,struct,subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import seal
ROOT=Path(__file__).resolve().parent;S=2**46

class Native:
    def __init__(self):
        src=ROOT/'modadd.c';out=ROOT/'modadd.so'
        if not out.exists() or out.stat().st_mtime<src.stat().st_mtime:
            subprocess.run(['cc','-O3','-shared','-fPIC',str(src),'-o',str(out)],check=True)
        self.lib=ctypes.CDLL(str(out));self.fn=self.lib.repeat_add
        self.fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*5+[ctypes.c_size_t]*4;self.fn.restype=ctypes.c_int

    def run(self,mode,a,b,mod,bias,out,repeats=1,tiles=1):
        if a.shape!=b.shape or a.shape!=out.shape or a.ndim!=4 or a.shape[0]!=tiles or a.shape[1]!=2:
            raise ValueError('Invalid residue array shape')
        _,_,limbs,n=a.shape
        if mod.shape!=(limbs,) or bias.shape!=(limbs,) or n%8:raise ValueError('Invalid modulus or length')
        if any(x.dtype!=np.uint64 or not x.flags.c_contiguous for x in (a,b,mod,bias,out)):
            raise ValueError('Expected contiguous uint64 arrays')
        if not out.flags.writeable or any(np.shares_memory(out,x) for x in (a,b,mod,bias)):
            raise ValueError('Output must be writable and disjoint')
        if not self.fn(mode,*(x.ctypes.data for x in (a,b,mod,bias,out)),n,limbs,repeats,tiles):
            raise ValueError('Native kernel rejected parameters or unavailable ISA')
        return out

@dataclass(frozen=True)
class RawCipher:
    prefix: bytes
    head: object
    tail: object
    moduli: object
    scale: float
    params_id: bytes

    def serialize(self,head=None):
        # Count every byte of the full reply; the common tail is not omitted.
        h=self.head if head is None else head
        return self.prefix+h.tobytes()+self.tail.tobytes()

def import_cipher(ct,context):
    raw=ct.to_string(seal.compr_mode_type.none)
    if len(raw)<113 or raw[:2]!=bytes.fromhex('5ea1') or raw[2]!=16 or raw[5]!=0 or ct.contains_seed():
        raise ValueError('Unsupported SEAL serialization')
    ntt,components,n,limbs,scale,correction=struct.unpack_from('<BQQQdQ',raw,48)
    count=struct.unpack_from('<Q',raw,105)[0]
    if (ntt,components,n,limbs,scale,correction)!=(1,3,8192,2,float(S*S),1) or count!=components*n*limbs or len(raw)!=113+8*count:
        raise ValueError('Only the tested full-level three-component CKKS layout is supported')
    if raw[89:91]!=bytes.fromhex('5ea1') or raw[94]!=0:raise ValueError('Unexpected nested array encoding')
    mod=np.array([p.value() for p in context.get_context_data(ct.parms_id()).parms().coeff_modulus()],dtype=np.uint64)
    values=np.frombuffer(raw,dtype='<u8',offset=113).reshape(3,limbs,n).copy()
    if np.any(values>=mod[None,:,None]):raise ValueError('Noncanonical ciphertext residues')
    head=np.ascontiguousarray(values[:2][None]);tail=np.ascontiguousarray(values[2:]);head.flags.writeable=False;tail.flags.writeable=False;mod.flags.writeable=False
    return RawCipher(raw[:113],head,tail,mod,scale,raw[16:48])

class RawDirectionServer:
    def __init__(self,context,base,direction,base_k,direction_k,mode=None):
        self.base=import_cipher(base,context);self.delta=import_cipher(direction,context)
        if self.base.params_id!=self.delta.params_id or not np.array_equal(self.base.moduli,self.delta.moduli):raise ValueError('Different contexts')
        if np.any(self.delta.tail):raise ValueError('Direction tail is not zero')
        self.base_k=np.asarray(base_k,dtype=np.int64).copy();self.direction_k=np.asarray(direction_k,dtype=np.int64).copy()
        self.native=Native()
        self.mode=(2 if self.native.lib.has_avx512() else 1 if self.native.lib.has_avx2() else 0) if mode is None else mode

    def reply(self,q):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(32,) or not np.all(np.isfinite(q)) or np.any(np.abs(q)>.5):raise ValueError('Invalid public query')
        k=np.rint(-2*S*q).astype(np.int64)
        if not np.array_equal(k,self.base_k+self.direction_k):raise ValueError('Direction certificate rejected')
        # SEAL scalar encoding at scale 2^92, rounded as binary64 before
        # conversion into each RNS prime. Whole-ciphertext equality verifies it.
        constant=int(round(float(q@q)*float(S*S)))
        bias=np.array([constant%int(p) for p in self.base.moduli],dtype=np.uint64)
        out=np.empty_like(self.base.head)
        self.native.run(self.mode,self.base.head,self.delta.head,self.base.moduli,bias,out)
        return self.base.serialize(out)
