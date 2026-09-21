# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Serialize into reusable server storage; return one independently owned bytes object."""
import ctypes
import numpy as np
from fresh_backend import FreshBackend,WORDS
class PackedBackend(FreshBackend):
 def __init__(self,fixture,gpu=False):
  super().__init__(fixture,gpu)
  self.reply_capacity=len(self.prefix)+WORDS*8
  self.reply_storage=ctypes.create_string_buffer(self.reply_capacity+63)
  # Align the payload despite the 113-byte wire header. No unaligned uint64 stores.
  payload=(ctypes.addressof(self.reply_storage)+len(self.prefix)+63)&~63
  self.reply_start=payload-len(self.prefix);self.reply_payload=ctypes.c_void_p(payload)
  ctypes.memmove(self.reply_start,self.prefix,len(self.prefix))
 def reply_packed(self,q):
  q=np.ascontiguousarray(q,dtype=np.float64)
  if q.shape!=(32,):raise ValueError('Expected 32 public coordinates')
  if not self.lib.fresh_reply(self.handle,q.ctypes.data,32,self.reply_payload,WORDS,2 if self.gpu else 1,0,None):raise ValueError(self.lib.fresh_error().decode())
  # Copies the complete canonical header and payload into new immutable storage.
  return ctypes.string_at(self.reply_start,self.reply_capacity)
