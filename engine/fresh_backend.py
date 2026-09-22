# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Server-only CPU/CUDA backends. No secret key crosses this API."""
import ctypes
from pathlib import Path
import numpy as np
import seal
ROOT=Path(__file__).resolve().parent
N=8192;WORDS=3*2*N

class FreshBackend:
 def __init__(self,fixture,gpu=False):
  self.gpu=gpu;self.lib=ctypes.CDLL(str(ROOT/('fresh_cuda.so' if gpu else 'fresh_cpu.so')))
  self.lib.fresh_error.restype=ctypes.c_char_p
  self.lib.fresh_create.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];self.lib.fresh_create.restype=ctypes.c_void_p
  self.lib.fresh_destroy.argtypes=[ctypes.c_void_p];self.lib.fresh_destroy.restype=None
  self.lib.fresh_reply.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];self.lib.fresh_reply.restype=ctypes.c_int
  self.lib.fresh_set_warm.argtypes=[ctypes.c_void_p]*3;self.lib.fresh_set_warm.restype=ctypes.c_int
  self.lib.fresh_plain.argtypes=[ctypes.c_void_p]*4;self.lib.fresh_plain.restype=ctypes.c_int
  srv=fixture['server'];context=fixture['context'];norm=srv.norm
  assert norm.size()==3 and norm.coeff_modulus_size()==2 and norm.is_ntt_form() and norm.scale()==2**92
  blob=norm.to_string(seal.compr_mode_type.none);self.prefix=blob[:113]
  nv=np.frombuffer(blob,dtype='<u8',offset=113);assert nv.size==WORDS
  cols=[]
  for ct in srv.encrypted:
   assert ct.size()==2 and ct.coeff_modulus_size()==2 and ct.is_ntt_form() and ct.scale()==2**46 and ct.parms_id()==norm.parms_id()
   data=ct.to_string(seal.compr_mode_type.none);arr=np.frombuffer(data,dtype='<u8',offset=113);assert arr.size==2*2*N;cols.append(arr)
  features=np.ascontiguousarray(cols,dtype=np.uint64)
  mod=np.array([p.value() for p in context.first_context_data().parms().coeff_modulus()],dtype=np.uint64)
  self.handle=self.lib.fresh_create(features.ctypes.data,features.size,nv.ctypes.data,nv.size,mod.ctypes.data)
  if not self.handle:raise ValueError(self.lib.fresh_error().decode())
  self.warm_k=None
 def set_warm(self,base,delta,q):
  b=base.to_string(seal.compr_mode_type.none);d=delta.to_string(seal.compr_mode_type.none)
  assert b[:113]==self.prefix and d[:113]==self.prefix
  bv=np.frombuffer(b,dtype='<u8',offset=113);dv=np.frombuffer(d,dtype='<u8',offset=113)
  if not self.lib.fresh_set_warm(self.handle,bv.ctypes.data,dv.ctypes.data):raise ValueError(self.lib.fresh_error().decode())
  self.warm_k=np.rint(-2**47*np.asarray(q)).astype(np.int64)
 def reply(self,q,method=None,warm=False,profile=False):
  if method is None:method=2 if self.gpu else 1
  q=np.ascontiguousarray(q,dtype=np.float64)
  if q.shape!=(32,):raise ValueError('Expected 32 public coordinates')
  if warm and (self.warm_k is None or not np.array_equal(np.rint(-2**47*q).astype(np.int64),self.warm_k)):raise ValueError('Warm certificate failed')
  out=np.empty(WORDS,dtype=np.uint64);timing=ctypes.c_double()
  if not self.lib.fresh_reply(self.handle,q.ctypes.data,32,out.ctypes.data,out.size,method,int(warm),ctypes.byref(timing) if profile else None):raise ValueError(self.lib.fresh_error().decode())
  blob=self.prefix+out.tobytes()
  return (blob,timing.value) if profile else blob
 def close(self):
  if getattr(self,'handle',None):self.lib.fresh_destroy(self.handle);self.handle=None
 def __del__(self):self.close()

class PlainGPU:
 def __init__(self,lib,x,norm):
  self.lib=lib;lib.plain_gpu_create.argtypes=[ctypes.c_void_p]*2;lib.plain_gpu_create.restype=ctypes.c_void_p
  lib.plain_gpu_destroy.argtypes=[ctypes.c_void_p];lib.plain_gpu_destroy.restype=None
  lib.plain_gpu_reply.argtypes=[ctypes.c_void_p]*3;lib.plain_gpu_reply.restype=ctypes.c_int
  self.handle=lib.plain_gpu_create(x.ctypes.data,norm.ctypes.data)
  if not self.handle:raise ValueError(lib.fresh_error().decode())
 def reply(self,q):
  q=np.ascontiguousarray(q,dtype=np.float64);out=np.empty(4096)
  if q.shape!=(32,) or not self.lib.plain_gpu_reply(self.handle,q.ctypes.data,out.ctypes.data):raise ValueError(self.lib.fresh_error().decode())
  return out
 def close(self):
  if getattr(self,'handle',None):self.lib.plain_gpu_destroy(self.handle);self.handle=None
 def __del__(self):self.close()
