// Copyright 2026 IPOQ, LLC
// SPDX-License-Identifier: Apache-2.0
// Specialized CKKS client, using SEAL validation and independent decode arithmetic.
// Only the client constructor accepts a secret key. No key is sent to the server.
#include <seal/seal.h>
#include <seal/util/ntt.h>
#include <seal/util/uintarithsmallmod.h>
#ifdef SEAL_USE_INTEL_HEXL
#include <hexl/eltwise/eltwise-mult-mod.hpp>
#include <hexl/eltwise/eltwise-fma-mod.hpp>
#include <hexl/eltwise/eltwise-reduce-mod.hpp>
#include <immintrin.h>
#endif
#define POCKETFFT_CACHE_SIZE 4
#define POCKETFFT_NO_MULTITHREADING
#include "vendor/pocketfft_hdronly.h"
#include <complex>
#include <cstring>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <array>

using namespace seal;
using namespace seal::util;
using u128=unsigned __int128;
constexpr size_t N=8192, SLOTS=N/2;
static thread_local std::string error_text;
static double clock_us(){return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();}

#ifdef SEAL_USE_INTEL_HEXL
// Reconstruct z=p0*t+a in radix 2^52, then center exactly modulo Q.
// Preconditions: canonical a<p0, t<p1, p1<2^52, p0*p1<2^104.
// Both centered digits are exactly convertible to double; their sum is rounded once.
template<bool Verify=false>
__attribute__((target("avx512f,avx512dq,avx512ifma")))
static bool crt52(const uint64_t*a,const uint64_t*t,uint64_t p0,uint64_t p1,double scale,double*out) {
    constexpr uint64_t mask=(uint64_t(1)<<52)-1;
    const u128 Q=u128(p0)*p1,H=Q/2;
    auto M=_mm512_set1_epi64(mask),Plo=_mm512_set1_epi64(p0&mask),Phi=_mm512_set1_epi64(p0>>52);
    auto Qlo=_mm512_set1_epi64(uint64_t(Q)&mask),Qhi=_mm512_set1_epi64(uint64_t(Q>>52));
    auto Hlo=_mm512_set1_epi64(uint64_t(H)&mask),Hhi=_mm512_set1_epi64(uint64_t(H>>52));
    auto zero=_mm512_setzero_si512(),sign=_mm512_set1_epi64(uint64_t(1)<<63);
    for(size_t i=0;i<N;i+=8) {
        auto A=_mm512_loadu_si512(a+i),T=_mm512_loadu_si512(t+i);
        auto lo=_mm512_add_epi64(_mm512_madd52lo_epu64(zero,Plo,T),_mm512_and_si512(A,M));
        auto hi=_mm512_add_epi64(_mm512_madd52hi_epu64(zero,Plo,T),_mm512_mullo_epi64(Phi,T));
        hi=_mm512_add_epi64(hi,_mm512_srli_epi64(A,52));hi=_mm512_add_epi64(hi,_mm512_srli_epi64(lo,52));lo=_mm512_and_si512(lo,M);
        if constexpr(Verify) {
            alignas(64) uint64_t lows[8],highs[8];_mm512_store_si512(lows,lo);_mm512_store_si512(highs,hi);
            for(size_t j=0;j<8;j++)if((u128(highs[j])<<52)+lows[j]!=u128(p0)*t[i+j]+a[i+j])return false;
        }
        auto negative=_mm512_cmp_epu64_mask(hi,Hhi,_MM_CMPINT_GT)|(_mm512_cmp_epu64_mask(hi,Hhi,_MM_CMPINT_EQ)&_mm512_cmp_epu64_mask(lo,Hlo,_MM_CMPINT_GT));
        auto borrow=_mm512_cmp_epu64_mask(Qlo,lo,_MM_CMPINT_LT);
        auto mlo=_mm512_and_si512(_mm512_sub_epi64(Qlo,lo),M);
        auto mhi=_mm512_mask_sub_epi64(_mm512_sub_epi64(Qhi,hi),borrow,_mm512_sub_epi64(Qhi,hi),_mm512_set1_epi64(1));
        lo=_mm512_mask_blend_epi64(negative,lo,mlo);hi=_mm512_mask_blend_epi64(negative,hi,mhi);
        // Separate exact power-of-two scaling and addition, avoiding 128-bit conversion helpers.
        auto v=_mm512_add_pd(_mm512_mul_pd(_mm512_cvtepu64_pd(hi),_mm512_set1_pd(0x1p52)),_mm512_cvtepu64_pd(lo));
        v=_mm512_castsi512_pd(_mm512_xor_si512(_mm512_castpd_si512(v),_mm512_maskz_mov_epi64(negative,sign)));
        v=_mm512_mul_pd(v,_mm512_set1_pd(scale));_mm512_storeu_pd(out+i,v);
        if constexpr(Verify)for(size_t j=0;j<8;j++){
            u128 z=u128(p0)*t[i+j]+a[i+j];double reference=(z>H?-double(Q-z):double(z))*scale;
            if(out[i+j]!=reference)return false;
        }
    }
    return true;
}
static bool supports_crt52(uint64_t p0,uint64_t p1) {
    return p1<(uint64_t(1)<<52) && (u128(p0)*p1>>104)==0 &&
           __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512ifma");
}
__attribute__((target("avx512f")))
static bool checked_copy512(const uint8_t*src,uint64_t*dst,uint64_t p) {
    auto modulus=_mm512_set1_epi64(p);__mmask8 invalid=0;
    for(size_t i=0;i<N;i+=8) {
        auto v=_mm512_loadu_si512(src+i*sizeof(uint64_t));
        invalid|=_mm512_cmp_epu64_mask(v,modulus,_MM_CMPINT_NLT);
        _mm512_storeu_si512(dst+i,v);
    }
    return !invalid;
}
#endif

struct Client {
    std::unique_ptr<SEALContext> context;
    std::unique_ptr<Decryptor> decryptor;
    std::unique_ptr<CKKSEncoder> encoder;
    Ciphertext ciphertext;
    Plaintext plain;
    std::vector<double> decoded;
    std::vector<uint64_t> coefficients;
    std::vector<std::complex<double>> values,twist;
    std::vector<size_t> slots,half_slots;
    std::vector<double> real_coefficients;
    std::vector<MultiplyUIntModOperand> sk1,sk2;
    std::vector<uint64_t> cached_tail,cached_product;
    std::vector<uint64_t> raw_sk1,raw_sk2,vector_product;
    std::vector<uint64_t> crt_work;
    parms_id_type cached_parms=parms_id_zero;
    uint64_t tail_hits=0,tail_misses=0;
    std::array<double,4> decode_phases{};
    std::array<double,6> detailed_decode_phases{};
    struct ImportShape {std::array<uint8_t,113> prefix;parms_id_type id;double scale;size_t limbs,bytes;};
    std::vector<ImportShape> import_shapes;
    uint64_t fast_imports=0,fallback_imports=0;

    Client(const uint8_t* p,size_t pn,const uint8_t* k,size_t kn)
        :coefficients(2*N),values(N),twist(N),slots(SLOTS),half_slots(SLOTS),real_coefficients(N),sk1(2*N),sk2(2*N),cached_tail(2*N),cached_product(2*N),raw_sk1(2*N),raw_sk2(2*N),vector_product(2*N),crt_work(N) {
        EncryptionParameters parms;parms.load(reinterpret_cast<const seal_byte*>(p),pn);
        context=std::make_unique<SEALContext>(parms,true,sec_level_type::tc128);
        if(!context->parameters_set() || parms.scheme()!=scheme_type::ckks || parms.poly_modulus_degree()!=N)
            throw std::invalid_argument("Unsupported or invalid client parameters");
        auto &mod=context->first_context_data()->parms().coeff_modulus();
        if(mod.size()!=2 || mod[0].bit_count()>60 || mod[1].bit_count()>60)
            throw std::invalid_argument("Expected two small data moduli");
        SecretKey key;key.load(*context,reinterpret_cast<const seal_byte*>(k),kn);
        decryptor=std::make_unique<Decryptor>(*context,key);
        encoder=std::make_unique<CKKSEncoder>(*context);
        for(size_t j=0;j<2;j++)for(size_t i=0;i<N;i++) {
            auto v=key.data()[j*N+i];sk1[j*N+i].set(v,mod[j]);
            sk2[j*N+i].set(multiply_uint_mod(v,v,mod[j]),mod[j]);
            raw_sk1[j*N+i]=v;raw_sk2[j*N+i]=multiply_uint_mod(v,v,mod[j]);
        }
        auto pi=std::acos(-1.0);
        for(size_t i=0;i<N;i++)twist[i]=std::polar(1.0,pi*double(i)/N);
        size_t pos=1;
        for(size_t i=0;i<SLOTS;i++){
            slots[i]=(pos-1)/2;
            // Roots congruent to 3 modulo 4 are conjugates of roots congruent to 1.
            half_slots[i]=((pos%4==1?pos:2*N-pos)-1)/4;
            pos=(3*pos)%(2*N);
        }
        auto cd=context->first_context_data();double scale=0x1p92;
        while(cd) {
            size_t limbs=cd->parms().coeff_modulus().size();
            if(std::isnormal(scale) && scale>0 && std::log2(scale)<cd->total_coeff_modulus_bit_count()) {
                Ciphertext canonical;canonical.resize(*context,cd->parms_id(),3);canonical.is_ntt_form()=true;canonical.scale()=scale;
                if(!is_valid_for(canonical,*context))throw std::logic_error("Invalid import template");
                size_t bytes=113+3*limbs*N*sizeof(uint64_t);
                if(size_t(canonical.save_size(compr_mode_type::none))!=bytes)throw std::logic_error("Unsupported serialization layout");
                std::vector<seal_byte> serialized(bytes);canonical.save(serialized.data(),serialized.size(),compr_mode_type::none);
                ImportShape shape{};std::memcpy(shape.prefix.data(),serialized.data(),113);shape.id=cd->parms_id();shape.scale=scale;shape.limbs=limbs;shape.bytes=bytes;import_shapes.push_back(shape);
            }
            scale/=double(cd->parms().coeff_modulus().back().value());cd=cd->next_context_data();
        }
        ciphertext.reserve(*context,context->first_parms_id(),3);
    }

    ~Client() {
        // Clear our copies of key-dependent preconditions as well as SEAL's own key object.
        for(auto *array:{&sk1,&sk2}) {
            volatile uint64_t *p=reinterpret_cast<volatile uint64_t*>(array->data());
            for(size_t i=0;i<array->size()*sizeof(MultiplyUIntModOperand)/sizeof(uint64_t);i++)p[i]=0;
        }
        for(auto *array:{&cached_product,&raw_sk1,&raw_sk2,&vector_product}) {
            volatile uint64_t *p=array->data();
            for(size_t i=0;i<array->size();i++)p[i]=0;
        }
    }

    void load(const uint8_t* bytes,size_t count,bool fast=false) {
        if(count>1024*1024)throw std::invalid_argument("Reply exceeds supported size");
        if(fast)for(const auto &shape:import_shapes)if(count==shape.bytes && std::memcmp(bytes,shape.prefix.data(),113)==0) {
            // The complete metadata comes from this client's trusted context, never from input sizes.
            ciphertext.resize(*context,shape.id,3);ciphertext.is_ntt_form()=true;ciphertext.scale()=shape.scale;ciphertext.correction_factor()=1;
            const auto &mod=context->get_context_data(shape.id)->parms().coeff_modulus();bool valid=true;
            for(size_t c=0;c<3;c++)for(size_t l=0;l<shape.limbs;l++) {
                size_t offset=(c*shape.limbs+l)*N;const auto*src=bytes+113+offset*sizeof(uint64_t);auto*dst=ciphertext.data()+offset;auto p=mod[l].value();
#ifdef SEAL_USE_INTEL_HEXL
                if(__builtin_cpu_supports("avx512f")){valid=checked_copy512(src,dst,p)&&valid;continue;}
#endif
                for(size_t i=0;i<N;i++){uint64_t v;std::memcpy(&v,src+i*sizeof(uint64_t),sizeof(v));dst[i]=v;valid=(v<p)&&valid;}
            }
            if(!valid)throw std::invalid_argument("Noncanonical ciphertext coefficient");
            ++fast_imports;return;
        }
        if(fast)++fallback_imports;
        if(size_t(ciphertext.load(*context,reinterpret_cast<const seal_byte*>(bytes),count))!=count)throw std::invalid_argument("Trailing bytes");
        if(ciphertext.size()!=3 || !ciphertext.is_ntt_form() || ciphertext.poly_modulus_degree()!=N ||
           ciphertext.coeff_modulus_size()<1 || ciphertext.coeff_modulus_size()>2)
            throw std::invalid_argument("Unsupported ciphertext layout");
        if(!std::isnormal(ciphertext.scale()) || ciphertext.scale()<=0 ||
           std::log2(ciphertext.scale())>=context->get_context_data(ciphertext.parms_id())->total_coeff_modulus_bit_count())
            throw std::invalid_argument("Invalid scale");
    }

    void preconditioned_decrypt(bool reuse_tail=false) {
        auto &mod=context->get_context_data(ciphertext.parms_id())->parms().coeff_modulus();
        size_t limbs=mod.size();
        // Called only after safe full-message import. Cache identity uses every
        // c2 byte plus its parameter ID, and is scoped to this client's key.
        // Scale is immaterial to this exact modular product.
        if(reuse_tail) {
            bool hit=cached_parms==ciphertext.parms_id() &&
                std::memcmp(cached_tail.data(),ciphertext.data(2),N*limbs*sizeof(uint64_t))==0;
            if(hit)++tail_hits;
            else {
                ++tail_misses;
                for(size_t j=0;j<limbs;j++)for(size_t i=0;i<N;i++)
                    cached_product[j*N+i]=multiply_uint_mod(ciphertext.data(2)[j*N+i],sk2[j*N+i],mod[j]);
                std::memcpy(cached_tail.data(),ciphertext.data(2),N*limbs*sizeof(uint64_t));
                cached_parms=ciphertext.parms_id();
            }
        }
        plain.parms_id()=parms_id_zero;plain.resize(N*limbs);
        for(size_t j=0;j<limbs;j++) {
            auto p=mod[j].value();size_t offset=j*N;
            const auto *c0=ciphertext.data(0)+offset,*c1=ciphertext.data(1)+offset,*c2=ciphertext.data(2)+offset;
            auto *destination=plain.data()+offset;
            const auto *s1=sk1.data()+offset,*s2=sk2.data()+offset;
            for(size_t i=0;i<N;i++) {
                uint64_t a=c0[i];uint64_t b=multiply_uint_mod(c1[i],s1[i],mod[j]);
                uint64_t c=reuse_tail?cached_product[offset+i]:multiply_uint_mod(c2[i],s2[i],mod[j]);
                uint64_t t=a+b;if(t>=p)t-=p;t+=c;if(t>=p)t-=p;destination[i]=t;
            }
        }
        plain.parms_id()=ciphertext.parms_id();plain.scale()=ciphertext.scale();
    }

    void fast_decode(double* output,bool half=false,bool detailed=false,int variant=0) {
        double t0=clock_us();
        bool constructed=variant==3;
        auto cd=context->get_context_data(constructed?ciphertext.parms_id():plain.parms_id());
        auto &mod=cd->parms().coeff_modulus();size_t limbs=mod.size();
        if(!constructed && (!is_valid_for(plain,*context) || !plain.is_ntt_form() || limbs<1 || limbs>2))
            throw std::invalid_argument("Invalid plaintext for specialized decoder");
        double validated=detailed?clock_us():0;
        if(!constructed)std::memcpy(coefficients.data(),plain.data(),N*limbs*sizeof(uint64_t));
        double copied=detailed?clock_us():0;
        for(size_t j=0;j<limbs;j++)inverse_ntt_negacyclic_harvey(coefficients.data()+j*N,cd->small_ntt_tables()[j]);
        double t1=clock_us();
        const uint64_t p0=mod[0].value();double inverse_scale=1.0/(constructed?ciphertext.scale():plain.scale());
        if(limbs==1) {
            for(size_t i=0;i<N;i++) {
                auto a=coefficients[i];double v=a>p0/2?-double(p0-a):double(a);
                if(half)real_coefficients[i]=v*inverse_scale;else values[i]=(v*inverse_scale)*twist[i];
            }
        } else {
            const uint64_t p1=mod[1].value();const u128 modulus=u128(p0)*p1;
            uint64_t inverse;
            if(!try_invert_uint_mod(p0%p1,mod[1],inverse))throw std::logic_error("CRT inverse failed");
            MultiplyUIntModOperand inv;inv.set(inverse,mod[1]);
            bool vector_crt=false;
#ifdef SEAL_USE_INTEL_HEXL
            vector_crt=variant>=2 && half && supports_crt52(p0,p1) && u128(p0)<=u128(p1)*p1;
            if(vector_crt) {
                intel::hexl::EltwiseReduceMod(crt_work.data(),coefficients.data(),N,p1,p1,1);
                for(size_t i=0;i<N;i++) {
                    auto a1=crt_work[i],b=coefficients[N+i];crt_work[i]=b>=a1?b-a1:b+p1-a1;
                }
                intel::hexl::EltwiseFMAMod(crt_work.data(),crt_work.data(),inverse,nullptr,N,p1,1);
                crt52(coefficients.data(),crt_work.data(),p0,p1,inverse_scale,real_coefficients.data());
            }
#endif
            if(!vector_crt)for(size_t i=0;i<N;i++) {
                auto a=coefficients[i],b=coefficients[N+i];auto a1=barrett_reduce_64(a,mod[1]);
                uint64_t difference=b>=a1?b-a1:b+p1-a1;
                u128 z=u128(p0)*multiply_uint_mod(difference,inv,mod[1])+a;
                // Center in integer arithmetic before conversion, avoiding floating cancellation.
                double v;
                if(variant==1) {
                    bool negative=z>modulus/2;u128 magnitude=negative?modulus-z:z;
                    v=double(uint64_t(magnitude>>64))*0x1p64+double(uint64_t(magnitude));
                    if(negative)v=-v;
                } else v=z>modulus/2?-double(modulus-z):double(z);
                if(half)real_coefficients[i]=v*inverse_scale;else values[i]=(v*inverse_scale)*twist[i];
            }
        }
        // P(z), z^N=-1. For z=zeta^(1+4k), z^(N/2)=i, so pair
        // a_j + i*a_(j+N/2), twist by zeta^j, and use an N/2 transform.
        if(half)for(size_t i=0;i<SLOTS;i++)values[i]=std::complex<double>(real_coefficients[i],real_coefficients[i+SLOTS])*twist[i];
        double t2=clock_us();
        pocketfft::c2c<double>({half?SLOTS:N},{sizeof(std::complex<double>)},{sizeof(std::complex<double>)},{0},
                              false,values.data(),values.data(),1.0,1);
        double t3=clock_us();
        for(size_t i=0;i<SLOTS;i++)output[i]=values[half?half_slots[i]:slots[i]].real();
        decode_phases={t1-t0,t2-t1,t3-t2,clock_us()-t3};
        if(detailed)detailed_decode_phases={validated-t0,copied-validated,t1-copied,t2-t1,t3-t2,decode_phases[3]};
    }

    void vector_decrypt(bool direct=false) {
#ifdef SEAL_USE_INTEL_HEXL
        auto &mod=context->get_context_data(ciphertext.parms_id())->parms().coeff_modulus();
        if(!direct){plain.parms_id()=parms_id_zero;plain.resize(N*mod.size());}
        for(size_t j=0;j<mod.size();j++) {
            size_t offset=j*N;auto p=mod[j].value();auto *destination=(direct?coefficients.data():plain.data())+offset;
            auto *extra=vector_product.data()+offset;const auto *c0=ciphertext.data(0)+offset;
            intel::hexl::EltwiseMultMod(destination,ciphertext.data(1)+offset,raw_sk1.data()+offset,N,p,1);
            intel::hexl::EltwiseMultMod(extra,ciphertext.data(2)+offset,raw_sk2.data()+offset,N,p,1);
            for(size_t i=0;i<N;i++) {
                uint64_t t=c0[i]+destination[i];if(t>=p)t-=p;t+=extra[i];if(t>=p)t-=p;destination[i]=t;
            }
        }
        if(!direct){plain.parms_id()=ciphertext.parms_id();plain.scale()=ciphertext.scale();}
#else
        throw std::invalid_argument("Vector decryption requires the HEXL build");
#endif
    }

    void run(int mode,double *output) {
        if(mode<0 || mode>10)throw std::invalid_argument("Unknown client mode");
        if(mode>=6)vector_decrypt(mode==10);
        else if(mode==2 || mode==4 || mode==5)preconditioned_decrypt(mode==5);else decryptor->decrypt(ciphertext,plain);
        if(mode==0) {
            encoder->decode(plain,decoded);std::memcpy(output,decoded.data(),SLOTS*sizeof(double));
        } else fast_decode(output,mode>=3,false,mode==10?3:mode>=8?2:mode==7?1:0);
    }
};

// A separate public-key-only server object. Its API has no secret-key argument.
struct NativeServer {
    std::unique_ptr<SEALContext> context;
    std::unique_ptr<Evaluator> evaluator;
    std::unique_ptr<CKKSEncoder> encoder;
    Ciphertext base,direction,working,compressed;
    Plaintext bias;
    std::array<int64_t,32> expected;
    std::array<uint8_t,113> raw_prefix;
    NativeServer(const uint8_t *p,size_t pn,const uint8_t *b,size_t bn,const uint8_t *d,size_t dn,const int64_t *k) {
        EncryptionParameters parms;parms.load(reinterpret_cast<const seal_byte*>(p),pn);
        context=std::make_unique<SEALContext>(parms,true,sec_level_type::tc128);
        if(!context->parameters_set() || parms.scheme()!=scheme_type::ckks || parms.poly_modulus_degree()!=N)
            throw std::invalid_argument("Invalid native server parameters");
        base.load(*context,reinterpret_cast<const seal_byte*>(b),bn);
        direction.load(*context,reinterpret_cast<const seal_byte*>(d),dn);
        if(base.size()!=3 || direction.size()!=3 || base.parms_id()!=direction.parms_id() ||
           base.coeff_modulus_size()!=2 || base.scale()!=std::ldexp(1.,92) || direction.scale()!=base.scale() ||
           !base.is_ntt_form() || !direction.is_ntt_form())throw std::invalid_argument("Invalid cached layout");
        for(size_t i=0;i<2*N;i++)if(direction.data(2)[i])throw std::invalid_argument("Nonzero direction tail");
        if(bn!=113+3*2*N*sizeof(uint64_t) || b[0]!=0x5e || b[1]!=0xa1 || b[5]!=0)
            throw std::invalid_argument("Unsupported uncompressed serialization");
        std::copy(b,b+113,raw_prefix.begin());
        evaluator=std::make_unique<Evaluator>(*context);encoder=std::make_unique<CKKSEncoder>(*context);
        std::copy(k,k+32,expected.begin());working=base;
    }
    size_t reply(const double *q,bool rescale,uint8_t *output,size_t capacity) {
        double norm=0;
        for(size_t j=0;j<32;j++) {
            if(!std::isfinite(q[j]) || std::abs(q[j])>.5 || std::nearbyint(-std::ldexp(1.,47)*q[j])!=double(expected[j]))
                throw std::invalid_argument("Public query certificate failed");
            norm+=q[j]*q[j];
        }
        auto &mod=context->get_context_data(base.parms_id())->parms().coeff_modulus();
        // Encoding a real scalar is constant in NTT form. Compute its two RNS
        // residues once instead of materializing a full scalar plaintext.
        u128 encoded_norm=static_cast<u128>(std::round(std::ldexp(norm,92)));
        // The unchanged third component stays in working and is still sent in full.
        for(size_t component=0;component<2;component++)for(size_t limb=0;limb<2;limb++) {
            auto p=mod[limb].value();auto scalar=uint64_t(encoded_norm%p);size_t offset=limb*N;
            const auto *a=base.data(component)+offset,*b=direction.data(component)+offset;
            auto *destination=working.data(component)+offset;
            for(size_t i=0;i<N;i++) {
                auto sum=a[i]+b[i];if(sum>=p)sum-=p;
                if(component==0){sum+=scalar;if(sum>=p)sum-=p;}
                destination[i]=sum;
            }
        }
        const Ciphertext *result=&working;
        if(!rescale){
            constexpr size_t bytes=113+3*2*N*sizeof(uint64_t);
            if(capacity<bytes)throw std::invalid_argument("Insufficient reply capacity");
            std::memcpy(output,raw_prefix.data(),113);std::memcpy(output+113,working.data(),bytes-113);return bytes;
        }
        if(rescale){evaluator->rescale_to_next(working,compressed);result=&compressed;}
        return size_t(result->save(reinterpret_cast<seal_byte*>(output),capacity,compr_mode_type::none));
    }
};

extern "C" {
#pragma GCC visibility push(default)
int plain_pipeline(const double*q,const int64_t*k,const double*base,const double*direction,double*out) {
    try {
        double bias=0;
        for(size_t j=0;j<32;j++){
            if(!std::isfinite(q[j]) || std::abs(q[j])>.5 || std::nearbyint(-std::ldexp(1.,47)*q[j])!=double(k[j]))
                throw std::invalid_argument("Plaintext certificate failed");
            bias+=q[j]*q[j];
        }
        std::vector<double> message(SLOTS);
        for(size_t i=0;i<SLOTS;i++)message[i]=base[i]+direction[i]+bias;
        std::memcpy(out,message.data(),SLOTS*sizeof(double));return 1;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
void* server_create(const uint8_t*p,size_t pn,const uint8_t*b,size_t bn,const uint8_t*d,size_t dn,const int64_t*k) {
    try{return new NativeServer(p,pn,b,bn,d,dn,k);}catch(const std::exception&e){error_text=e.what();return nullptr;}
}
void server_destroy(void*p){delete static_cast<NativeServer*>(p);}
size_t server_reply(void*h,const double*q,size_t dimensions,int rescale,uint8_t*out,size_t capacity) {
    try {
        if(!h || !q || !out || dimensions!=32 || (rescale!=0 && rescale!=1))throw std::invalid_argument("Invalid server buffers");
        return static_cast<NativeServer*>(h)->reply(q,rescale!=0,out,capacity);
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
const char* client_error(){return error_text.c_str();}
void* client_create(const uint8_t* p,size_t pn,const uint8_t* k,size_t kn) {
    try{return new Client(p,pn,k,kn);}catch(const std::exception&e){error_text=e.what();return nullptr;}
}
void client_destroy(void* p){delete static_cast<Client*>(p);}
int client_decode(void* h,const uint8_t* bytes,size_t count,int mode,double* output,size_t outputs) {
    try{
        if(!h || !bytes || !output || outputs!=SLOTS)throw std::invalid_argument("Invalid client buffers");
        auto &c=*static_cast<Client*>(h);c.load(bytes,count,mode>=9);c.run(mode,output);return 1;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_verify_decryption(void* h,const uint8_t* bytes,size_t count) {
    try{
        auto &c=*static_cast<Client*>(h);c.load(bytes,count);Plaintext reference;
        c.decryptor->decrypt(c.ciphertext,reference);c.preconditioned_decrypt();
        if(reference.coeff_count()!=c.plain.coeff_count() || reference.scale()!=c.plain.scale() ||
           reference.parms_id()!=c.plain.parms_id())return 0;
        return std::memcmp(reference.data(),c.plain.data(),reference.coeff_count()*sizeof(uint64_t))==0;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_verify_cached_decryption(void* h,const uint8_t* bytes,size_t count) {
    try{
        if(!h || !bytes)throw std::invalid_argument("Invalid verification buffers");
        auto &c=*static_cast<Client*>(h);c.load(bytes,count);Plaintext reference;
        c.decryptor->decrypt(c.ciphertext,reference);c.preconditioned_decrypt(true);
        return reference.coeff_count()==c.plain.coeff_count() && reference.scale()==c.plain.scale() &&
            reference.parms_id()==c.plain.parms_id() &&
            std::memcmp(reference.data(),c.plain.data(),reference.coeff_count()*sizeof(uint64_t))==0;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_tail_stats(void* h,uint64_t* out) {
    if(!h || !out)return 0;
    auto &c=*static_cast<Client*>(h);out[0]=c.tail_hits;out[1]=c.tail_misses;return 1;
}
int client_verify_vector_decryption(void* h,const uint8_t* bytes,size_t count) {
    try{
        if(!h || !bytes)throw std::invalid_argument("Invalid verification buffers");
        auto &c=*static_cast<Client*>(h);c.load(bytes,count);Plaintext reference;
        c.decryptor->decrypt(c.ciphertext,reference);c.vector_decrypt();
        return reference.coeff_count()==c.plain.coeff_count() && reference.scale()==c.plain.scale() &&
            reference.parms_id()==c.plain.parms_id() &&
            std::memcmp(reference.data(),c.plain.data(),reference.coeff_count()*sizeof(uint64_t))==0;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_profile(void* h,const uint8_t* bytes,size_t count,double* phases) {
    try{
        if(!h || !bytes || !phases)throw std::invalid_argument("Invalid profile buffers");
        auto &c=*static_cast<Client*>(h);std::vector<double> output(SLOTS);
        double a=clock_us();c.load(bytes,count);double b=clock_us();c.preconditioned_decrypt();double d=clock_us();
        c.fast_decode(output.data());phases[0]=b-a;phases[1]=d-b;
        std::copy(c.decode_phases.begin(),c.decode_phases.end(),phases+2);return 1;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_profile_current(void* h,const uint8_t* bytes,size_t count,int mode,double* output,double* phases) {
    try{
        if(!h || !bytes || !phases || !output || (mode!=4 && (mode<6 || mode>10)))throw std::invalid_argument("Invalid current-profile arguments");
        auto &c=*static_cast<Client*>(h);
        double a=clock_us();c.load(bytes,count,mode>=9);double b=clock_us();
        if(mode>=6)c.vector_decrypt(mode==10);else c.preconditioned_decrypt();
        double d=clock_us();c.fast_decode(output,true,true,mode==10?3:mode>=8?2:mode==7?1:0);
        phases[0]=b-a;phases[1]=d-b;
        std::copy(c.detailed_decode_phases.begin(),c.detailed_decode_phases.end(),phases+2);return 1;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_verify_crt52(const uint64_t*a,const uint64_t*t,size_t count,uint64_t p0,uint64_t p1) {
    try {
#ifdef SEAL_USE_INTEL_HEXL
        if(!a || !t || count!=N || p0<2 || p0>=(uint64_t(1)<<60) || p1<2 || !supports_crt52(p0,p1))throw std::invalid_argument("Unsupported CRT verification inputs");
        for(size_t i=0;i<N;i++)if(a[i]>=p0 || t[i]>=p1)throw std::invalid_argument("Noncanonical CRT verification inputs");
        std::vector<double> out(N);return crt52<true>(a,t,p0,p1,0x1p-92,out.data());
#else
        return 0;
#endif
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_verify_fast(void*h,const uint8_t*bytes,size_t count) {
    try {
        if(!h || !bytes)throw std::invalid_argument("Invalid verification buffers");
        auto &c=*static_cast<Client*>(h);c.load(bytes,count,true);
        if(!is_valid_for(c.ciphertext,*c.context))throw std::logic_error("Import validation mismatch");
        Plaintext ref;c.decryptor->decrypt(c.ciphertext,ref);c.vector_decrypt(true);
        if(ref.coeff_count()!=N*c.ciphertext.coeff_modulus_size())return 0;
        return std::memcmp(ref.data(),c.coefficients.data(),ref.coeff_count()*sizeof(uint64_t))==0;
    }catch(const std::exception&e){error_text=e.what();return 0;}
}
int client_import_stats(void*h,uint64_t*out) {
    if(!h || !out)return 0;auto &c=*static_cast<Client*>(h);out[0]=c.fast_imports;out[1]=c.fallback_imports;return 1;
}
}
#pragma GCC visibility pop
