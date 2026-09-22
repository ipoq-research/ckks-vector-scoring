// Copyright 2026 IPOQ, LLC
// SPDX-License-Identifier: Apache-2.0
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

static inline uint64_t addmod(uint64_t a,uint64_t b,uint64_t p){uint64_t s=a+b;return s>=p?s-p:s;}

/* Inputs are canonical residues and p<2^60. Therefore a+b cannot overflow. */
__attribute__((noinline,optimize("no-tree-vectorize")))
void scalar_add(const uint64_t*a,const uint64_t*b,const uint64_t*p,const uint64_t*bias,
                uint64_t*out,size_t n,size_t limbs){
    for(size_t c=0;c<2;c++)for(size_t l=0;l<limbs;l++)for(size_t j=0;j<n;j++){
        size_t i=(c*limbs+l)*n+j;uint64_t s=addmod(a[i],b[i],p[l]);
        out[i]=c==0?addmod(s,bias[l],p[l]):s;
    }
}

__attribute__((noinline,target("avx2")))
void avx2_add(const uint64_t*a,const uint64_t*b,const uint64_t*p,const uint64_t*bias,
             uint64_t*out,size_t n,size_t limbs){
    for(size_t c=0;c<2;c++)for(size_t l=0;l<limbs;l++){
        __m256i mod=_mm256_set1_epi64x(p[l]),lim=_mm256_set1_epi64x(p[l]-1),v=_mm256_set1_epi64x(bias[l]);
        for(size_t j=0;j<n;j+=4){size_t i=(c*limbs+l)*n+j;
            __m256i s=_mm256_add_epi64(_mm256_loadu_si256((const __m256i*)(a+i)),_mm256_loadu_si256((const __m256i*)(b+i)));
            s=_mm256_sub_epi64(s,_mm256_and_si256(_mm256_cmpgt_epi64(s,lim),mod));
            if(c==0){s=_mm256_add_epi64(s,v);s=_mm256_sub_epi64(s,_mm256_and_si256(_mm256_cmpgt_epi64(s,lim),mod));}
            _mm256_storeu_si256((__m256i*)(out+i),s);
        }
    }
}

__attribute__((noinline,target("avx512f")))
void avx512_add(const uint64_t*a,const uint64_t*b,const uint64_t*p,const uint64_t*bias,
               uint64_t*out,size_t n,size_t limbs){
    for(size_t c=0;c<2;c++)for(size_t l=0;l<limbs;l++){
        __m512i mod=_mm512_set1_epi64(p[l]),v=_mm512_set1_epi64(bias[l]);
        for(size_t j=0;j<n;j+=8){size_t i=(c*limbs+l)*n+j;
            __m512i s=_mm512_add_epi64(_mm512_loadu_si512(a+i),_mm512_loadu_si512(b+i));
            s=_mm512_mask_sub_epi64(s,_mm512_cmp_epu64_mask(s,mod,_MM_CMPINT_NLT),s,mod);
            if(c==0){s=_mm512_add_epi64(s,v);s=_mm512_mask_sub_epi64(s,_mm512_cmp_epu64_mask(s,mod,_MM_CMPINT_NLT),s,mod);}
            _mm512_storeu_si512(out+i,s);
        }
    }
}

int has_avx512(void){return __builtin_cpu_supports("avx512f")!=0;}
int has_avx2(void){return __builtin_cpu_supports("avx2")!=0;}
typedef void(*kernel)(const uint64_t*,const uint64_t*,const uint64_t*,const uint64_t*,uint64_t*,size_t,size_t);
int repeat_add(int mode,const uint64_t*a,const uint64_t*b,const uint64_t*p,const uint64_t*bias,
               uint64_t*out,size_t n,size_t limbs,size_t repeats,size_t tiles){
    if(!n||n%8||!limbs||!repeats||!tiles||mode<0||mode>2)return 0;
    if(mode==1&&!has_avx2())return 0;if(mode==2&&!has_avx512())return 0;
    for(size_t l=0;l<limbs;l++)if(p[l]>(1ULL<<60)||bias[l]>=p[l])return 0;
    kernel f=mode==2?avx512_add:mode==1?avx2_add:scalar_add;
    size_t stride=2*n*limbs;
    for(size_t r=0;r<repeats;r++){size_t offset=(r%tiles)*stride;f(a+offset,b+offset,p,bias,out+offset,n,limbs);__asm__ __volatile__("" ::: "memory");}
    return 1;
}

__attribute__((noinline,target("avx512f")))
void plain_add(const double*a,const double*b,double bias,double*out,size_t n,size_t repeats){
    for(size_t r=0;r<repeats;r++){
        for(size_t i=0;i<n;i+=8)_mm512_storeu_pd(out+i,_mm512_add_pd(_mm512_add_pd(_mm512_loadu_pd(a+i),_mm512_loadu_pd(b+i)),_mm512_set1_pd(bias)));
        __asm__ __volatile__("" ::: "memory");
    }
}
