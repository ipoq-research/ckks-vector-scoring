// Copyright 2026 IPOQ, LLC
// SPDX-License-Identifier: Apache-2.0
// Fixed-shape encrypted distance evaluation. Server API accepts no secret key.
// CPU and CUDA compute the same canonical modular operations, with full outputs.
#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#ifdef FHE_HEXL
#include <hexl/eltwise/eltwise-fma-mod.hpp>
#endif
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_TRY(call) do { auto e=(call);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }while(0)
#endif
using U=uint64_t;using W=unsigned __int128;
constexpr size_t N=8192,L=2,D=32,C=3,WORDS=C*L*N,FEATURE_WORDS=D*2*L*N;
static thread_local std::string error;
struct Query {U operand[L][D],quotient[L][D],bias[L],mod[L];};
Query prepare(const double*q,const U*p) {
 Query a{};double norm=0;
 for(size_t j=0;j<D;j++) {
  if(!std::isfinite(q[j]) || std::abs(q[j])>.5)throw std::invalid_argument("Query outside public domain");
  norm+=q[j]*q[j];int64_t k=static_cast<int64_t>(std::nearbyint(-std::ldexp(1.,47)*q[j]));
  for(size_t l=0;l<L;l++) {U v=U(k<0?-k:k)%p[l];if(k<0&&v)v=p[l]-v;a.operand[l][j]=v;a.quotient[l][j]=U((W(v)<<64)/p[l]);}
 }
 W b=static_cast<W>(std::round(std::ldexp(norm,92)));
 for(size_t l=0;l<L;l++){a.mod[l]=p[l];a.bias[l]=U(b%p[l]);}
 return a;
}
inline U mul(U x,U y,U quotient,U p){U q=U((W(x)*quotient)>>64);U z=x*y-q*p;return z>=p?z-p:z;}
inline U add(U x,U y,U p){U z=x+y;return z>=p?z-p:z;}
#ifdef __CUDACC__
__device__ __forceinline__ U dmul(U x,U y,U quotient,U p){U q=__umul64hi(x,quotient);U z=x*y-q*p;return z>=p?z-p:z;}
__global__ void fresh_kernel(const U*features,const U*norm,U*out,Query a) {
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=WORDS)return;
 size_t component=i/(L*N),limb=i/N%L;U p=a.mod[limb],v=norm[i];
 if(component<2) {
  if(component==0){v+=a.bias[limb];if(v>=p)v-=p;}
  for(size_t j=0;j<D;j++){U z=dmul(features[j*2*L*N+i],a.operand[limb][j],a.quotient[limb][j],p);v+=z;if(v>=p)v-=p;}
 }
 out[i]=v;
}
__global__ void warm_kernel(const U*a,const U*b,U*out,Query q) {
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=WORDS)return;
 size_t component=i/(L*N),limb=i/N%L;U p=q.mod[limb],v=a[i];
 if(component<2){v+=b[i];if(v>=p)v-=p;if(component==0){v+=q.bias[limb];if(v>=p)v-=p;}}
 out[i]=v;
}
#endif
struct Backend {
 std::vector<U> features,norm,warm_base,warm_delta;U mod[L];
#ifdef __CUDACC__
 U *gpu_features=nullptr,*gpu_norm=nullptr,*gpu_out=nullptr,*gpu_base=nullptr,*gpu_delta=nullptr,*pinned=nullptr;
 cudaEvent_t start{},end{};bool events=false;
#endif
 Backend(const U*f,const U*n,const U*p):features(f,f+FEATURE_WORDS),norm(n,n+WORDS) {
  for(size_t l=0;l<L;l++){if(p[l]<2||p[l]>=U(1)<<60)throw std::invalid_argument("Unsupported modulus");mod[l]=p[l];}
  for(size_t i=0;i<WORDS;i++)if(norm[i]>=mod[i/N%L])throw std::invalid_argument("Noncanonical norm");
  for(size_t i=0;i<FEATURE_WORDS;i++)if(features[i]>=mod[i/N%L])throw std::invalid_argument("Noncanonical feature");
#ifdef __CUDACC__
  try {
   CUDA_TRY(cudaMalloc(&gpu_features,FEATURE_WORDS*sizeof(U)));CUDA_TRY(cudaMalloc(&gpu_norm,WORDS*sizeof(U)));
   CUDA_TRY(cudaMalloc(&gpu_out,WORDS*sizeof(U)));CUDA_TRY(cudaMallocHost(&pinned,WORDS*sizeof(U)));
   CUDA_TRY(cudaMemcpy(gpu_features,f,FEATURE_WORDS*sizeof(U),cudaMemcpyHostToDevice));CUDA_TRY(cudaMemcpy(gpu_norm,n,WORDS*sizeof(U),cudaMemcpyHostToDevice));
   CUDA_TRY(cudaEventCreate(&start));CUDA_TRY(cudaEventCreate(&end));events=true;
  }catch(...){release();throw;}
#endif
 }
 void release(){
#ifdef __CUDACC__
  if(events){cudaEventDestroy(start);cudaEventDestroy(end);}cudaFree(gpu_features);cudaFree(gpu_norm);cudaFree(gpu_out);cudaFree(gpu_base);cudaFree(gpu_delta);cudaFreeHost(pinned);
#endif
 }
 ~Backend(){release();}
 void set_warm(const U*b,const U*d){
  for(size_t i=0;i<WORDS;i++)if(b[i]>=mod[i/N%L]||d[i]>=mod[i/N%L]||(i>=2*L*N&&d[i]))throw std::invalid_argument("Invalid warm cache component");
  warm_base.assign(b,b+WORDS);warm_delta.assign(d,d+WORDS);
#ifdef __CUDACC__
  if(!gpu_base)CUDA_TRY(cudaMalloc(&gpu_base,WORDS*sizeof(U)));if(!gpu_delta)CUDA_TRY(cudaMalloc(&gpu_delta,WORDS*sizeof(U)));
  CUDA_TRY(cudaMemcpy(gpu_base,b,WORDS*sizeof(U),cudaMemcpyHostToDevice));CUDA_TRY(cudaMemcpy(gpu_delta,d,WORDS*sizeof(U),cudaMemcpyHostToDevice));
#endif
 }
 void cpu(const Query&a,U*out,bool warm,int method) {
  auto &base=warm?warm_base:norm;if(warm&&base.empty())throw std::invalid_argument("Warm state absent");
  std::memcpy(out,base.data(),WORDS*sizeof(U));
  for(size_t c=0;c<2;c++)for(size_t l=0;l<L;l++) {
   U p=mod[l];size_t offset=(c*L+l)*N;U *dest=out+offset;
   if(c==0)for(size_t i=0;i<N;i++)dest[i]=add(dest[i],a.bias[l],p);
   if(warm){for(size_t i=0;i<N;i++)dest[i]=add(dest[i],warm_delta[offset+i],p);continue;}
   for(size_t j=0;j<D;j++) {
    const U*f=features.data()+j*2*L*N+offset;
#ifdef FHE_HEXL
    if(method==1){intel::hexl::EltwiseFMAMod(dest,f,a.operand[l][j],dest,N,p,1);continue;}
#endif
    for(size_t i=0;i<N;i++)dest[i]=add(dest[i],mul(f[i],a.operand[l][j],a.quotient[l][j],p),p);
   }
  }
 }
 void run(const double*q,U*out,int method,bool warm,double*kernel_us) {
  Query a=prepare(q,mod);
  if(method<2){cpu(a,out,warm,method);if(kernel_us)*kernel_us=-1;return;}
#ifdef __CUDACC__
  if(warm&&warm_base.empty())throw std::invalid_argument("Warm state absent");
  if(kernel_us)CUDA_TRY(cudaEventRecord(start));
  if(warm)warm_kernel<<<(WORDS+255)/256,256>>>(gpu_base,gpu_delta,gpu_out,a);
  else fresh_kernel<<<(WORDS+255)/256,256>>>(gpu_features,gpu_norm,gpu_out,a);
  CUDA_TRY(cudaGetLastError());if(kernel_us)CUDA_TRY(cudaEventRecord(end));
  CUDA_TRY(cudaMemcpy(pinned,gpu_out,WORDS*sizeof(U),cudaMemcpyDeviceToHost));
  std::memcpy(out,pinned,WORDS*sizeof(U));
  if(kernel_us){float ms;CUDA_TRY(cudaEventElapsedTime(&ms,start,end));*kernel_us=double(ms)*1000;}
#else
  throw std::invalid_argument("CUDA build required");
#endif
 }
};
extern "C" {
const char* fresh_error(){return error.c_str();}
void* fresh_create(const U*f,size_t nf,const U*n,size_t nn,const U*p) {
 try{if(!f||!n||!p||nf!=FEATURE_WORDS||nn!=WORDS)throw std::invalid_argument("Invalid layout");return new Backend(f,n,p);}catch(const std::exception&e){error=e.what();return nullptr;}
}
void fresh_destroy(void*h){delete static_cast<Backend*>(h);}
int fresh_set_warm(void*h,const U*b,const U*d){try{if(!h||!b||!d)throw std::invalid_argument("Invalid warm buffers");static_cast<Backend*>(h)->set_warm(b,d);return 1;}catch(const std::exception&e){error=e.what();return 0;}}
int fresh_reply(void*h,const double*q,size_t nq,U*out,size_t no,int method,int warm,double*kernel_us) {
 try{if(!h||!q||!out||nq!=D||no!=WORDS||method<0||method>2||(warm!=0&&warm!=1))throw std::invalid_argument("Invalid buffers");
  static_cast<Backend*>(h)->run(q,out,method,warm,kernel_us);return 1;}catch(const std::exception&e){error=e.what();return 0;}
}
int fresh_plain(const double*x,const double*norm,const double*q,double*out) {
 try{if(!x||!norm||!q||!out)throw std::invalid_argument("Invalid plaintext buffers");double bias=0;
  for(size_t j=0;j<D;j++){if(!std::isfinite(q[j])||std::abs(q[j])>.5)throw std::invalid_argument("Invalid query");bias+=q[j]*q[j];}
  std::vector<double>message(N/2);for(size_t i=0;i<N/2;i++)message[i]=norm[i]+bias;
  for(size_t j=0;j<D;j++)for(size_t i=0;i<N/2;i++)message[i]+=-2*q[j]*x[j*(N/2)+i];
  std::memcpy(out,message.data(),N/2*sizeof(double));return 1;
 }catch(const std::exception&e){error=e.what();return 0;}
}
int fresh_has_hexl(){
#ifdef FHE_HEXL
 return 1;
#else
 return 0;
#endif
}
}
#ifdef __CUDACC__
// Separate plaintext GPU control. Never used to evaluate encrypted replies.
struct PlainQuery{double q[D];double bias;};
__global__ void plain_kernel(const double*x,const double*norm,double*out,PlainQuery q) {
 size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N/2)return;
 double v=norm[i]+q.bias;for(size_t j=0;j<D;j++)v+=(-2*q.q[j])*x[j*(N/2)+i];out[i]=v;
}
struct PlainGPU {
 double *x=nullptr,*norm=nullptr,*out=nullptr,*pinned=nullptr;
 PlainGPU(const double*xx,const double*nn){try{
 CUDA_TRY(cudaMalloc(&x,D*(N/2)*sizeof(double)));CUDA_TRY(cudaMalloc(&norm,(N/2)*sizeof(double)));CUDA_TRY(cudaMalloc(&out,(N/2)*sizeof(double)));CUDA_TRY(cudaMallocHost(&pinned,(N/2)*sizeof(double)));
 CUDA_TRY(cudaMemcpy(x,xx,D*(N/2)*sizeof(double),cudaMemcpyHostToDevice));CUDA_TRY(cudaMemcpy(norm,nn,(N/2)*sizeof(double),cudaMemcpyHostToDevice));
 }catch(...){release();throw;}}
 void release(){cudaFree(x);cudaFree(norm);cudaFree(out);cudaFreeHost(pinned);}
 ~PlainGPU(){release();}
 void reply(const double*q,double*result){PlainQuery p{};for(size_t j=0;j<D;j++){if(!std::isfinite(q[j])||std::abs(q[j])>.5)throw std::invalid_argument("Invalid plaintext query");p.q[j]=q[j];p.bias+=q[j]*q[j];}
 plain_kernel<<<((N/2)+255)/256,256>>>(x,norm,out,p);CUDA_TRY(cudaGetLastError());CUDA_TRY(cudaMemcpy(pinned,out,(N/2)*sizeof(double),cudaMemcpyDeviceToHost));std::memcpy(result,pinned,(N/2)*sizeof(double));}
};
extern "C" {
void* plain_gpu_create(const double*x,const double*n){try{if(!x||!n)throw std::invalid_argument("Invalid plaintext inputs");return new PlainGPU(x,n);}catch(const std::exception&e){error=e.what();return nullptr;}}
void plain_gpu_destroy(void*h){delete static_cast<PlainGPU*>(h);}
int plain_gpu_reply(void*h,const double*q,double*out){try{if(!h||!q||!out)throw std::invalid_argument("Invalid buffers");static_cast<PlainGPU*>(h)->reply(q,out);return 1;}catch(const std::exception&e){error=e.what();return 0;}}
}
#endif
