# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""One native call from serialized encrypted reply to owned NumPy result."""
import ctypes
from pathlib import Path
import numpy as np
import seal

class NativeClient:
    def __init__(self,params,secret_key,hexl=False,library=None):
        self.hexl=hexl
        self.lib=ctypes.CDLL(str(Path(__file__).resolve().parent/(library or ('client_native_hexl.so' if hexl else 'client_native.so'))))
        self.lib.client_error.restype=ctypes.c_char_p
        self.lib.client_create.argtypes=[ctypes.c_void_p,ctypes.c_size_t]*2
        self.lib.client_create.restype=ctypes.c_void_p
        self.lib.client_decode.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.client_decode.restype=ctypes.c_int
        self.lib.client_verify_decryption.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.client_verify_decryption.restype=ctypes.c_int
        self.lib.client_verify_cached_decryption.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.client_verify_cached_decryption.restype=ctypes.c_int
        self.lib.client_tail_stats.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        self.lib.client_tail_stats.restype=ctypes.c_int
        self.lib.client_verify_vector_decryption.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.client_verify_vector_decryption.restype=ctypes.c_int
        self.lib.client_destroy.argtypes=[ctypes.c_void_p]
        self.lib.client_destroy.restype=None
        p=params.to_bytes(seal.compr_mode_type.none);k=secret_key.to_string()
        self.handle=self.lib.client_create(p,len(p),k,len(k))
        if not self.handle:raise ValueError(self.lib.client_error().decode())

    def decode(self,blob,mode=None,out=None):
        if mode is None:mode=4 if self.hexl else 2
        if not self.handle or not isinstance(blob,bytes):raise ValueError('Expected live client and immutable reply bytes')
        if out is None:out=np.empty(4096,dtype=np.float64)
        if out.shape!=(4096,) or out.dtype!=np.float64 or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('Expected writable contiguous 4096-double output')
        if not self.lib.client_decode(self.handle,blob,len(blob),mode,out.ctypes.data,len(out)):
            raise ValueError(self.lib.client_error().decode())
        return out

    def verify_decryption(self,blob):
        if not self.handle:raise ValueError('Client is closed')
        return bool(self.lib.client_verify_decryption(self.handle,blob,len(blob)))

    def verify_cached_decryption(self,blob):
        if not self.handle:raise ValueError('Client is closed')
        return bool(self.lib.client_verify_cached_decryption(self.handle,blob,len(blob)))

    def tail_stats(self):
        values=(ctypes.c_uint64*2)()
        if not self.lib.client_tail_stats(self.handle,values):raise ValueError('Client is closed')
        return dict(hits=values[0],misses=values[1])

    def verify_vector_decryption(self,blob):
        if not self.handle:raise ValueError('Client is closed')
        return bool(self.lib.client_verify_vector_decryption(self.handle,blob,len(blob)))

    def close(self):
        if getattr(self,'handle',None):self.lib.client_destroy(self.handle);self.handle=None

    def __del__(self):self.close()

class NativeServer:
    """Server-side native integration, constructed only from public parameters and ciphertexts."""
    def __init__(self,params,raw_server,hexl=True):
        self.lib=ctypes.CDLL(str(Path(__file__).resolve().parent/('client_native_hexl.so' if hexl else 'client_native.so')))
        self.lib.client_error.restype=ctypes.c_char_p
        self.lib.server_create.argtypes=[ctypes.c_void_p,ctypes.c_size_t]*3+[ctypes.c_void_p]
        self.lib.server_create.restype=ctypes.c_void_p
        self.lib.server_reply.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
        self.lib.server_reply.restype=ctypes.c_size_t
        self.lib.server_destroy.argtypes=[ctypes.c_void_p];self.lib.server_destroy.restype=None
        p=params.to_bytes(seal.compr_mode_type.none);b=raw_server.base.serialize();d=raw_server.delta.serialize()
        expected=np.ascontiguousarray(raw_server.base_k+raw_server.direction_k,dtype=np.int64)
        self.handle=self.lib.server_create(p,len(p),b,len(b),d,len(d),expected.ctypes.data)
        if not self.handle:raise ValueError(self.lib.client_error().decode())
        self.capacity=len(b)
        self.buffer=ctypes.create_string_buffer(self.capacity)
    def reply(self,q,rescale=False):
        q=np.asarray(q,dtype=np.float64)
        if not self.handle or q.shape!=(32,) or not q.flags.c_contiguous:raise ValueError('Invalid query shape or server')
        out=self.buffer
        count=self.lib.server_reply(self.handle,q.ctypes.data,32,int(rescale),out,self.capacity)
        if not count:raise ValueError(self.lib.client_error().decode())
        return ctypes.string_at(out,count)
    def close(self):
        if getattr(self,'handle',None):self.lib.server_destroy(self.handle);self.handle=None
    def __del__(self):self.close()
