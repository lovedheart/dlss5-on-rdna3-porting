// gfx1103 (RDNA3) replacements for gfx12 FP8/f16 WMMA + FP8 cvt builtins.
// Fragment convention (derived from gfx12 staging in the production family):
// lane l (rc=l&15, g=l>>4) holds fragment values at k' in [g*8,g*8+8) of
// row/col rc (fp8: 2 packed words; f16: 4 half-pair words); C: lane owns
// rows g*8+e (e=0..7) at column rc. Emulation stages fragments into a
// per-wave LDS slab, waits lgkmcnt (wave-local visibility, no barrier),
// then reads in the C pattern, k ascending (the validated MAC order).
// fp8 byte->float decode uses a 288-entry LDS table (bit-exact to sfp8v;
// position (b*9)%288 keeps lane strides bank-spread) - the bit-twiddle
// version blew the instruction cache (32K unpack ops per kernel body).
#pragma once
typedef unsigned uint;
#define R3DEV __attribute__((device)) __attribute__((always_inline))
namespace rdna3 {
R3DEV uint bits(float x){union{float f;uint u;}t;t.f=x;return t.u;}
R3DEV float val(uint u){union{float f;uint u;}t;t.u=u;return t.f;}
R3DEV uint hb16(_Float16 h){union{_Float16 h;unsigned short s;}t{h};return t.s;}
R3DEV uint sq8(float x){uint b=bits(x),a=b&0x7fffffffu,s=(b>>28)&8u;
 if(a>=0x7f800000u)return (s<<4)|0x7fu; if(a>=0x43e00000u)return (s<<4)|0x7eu;
 if(a<0x3c800000u){float sc=val(a)*512.f;uint q=uint(sc),r=bits(sc-float(q));q+=uint(r>0x3f000000u||(r==0x3f000000u&&(q&1u)));return (s<<4)|(q>7u?8u:q);}
 int e=int(a>>23)-127;uint m=a&0x7fffffu,k=m>>20,d=m&0xfffffu;uint rr=k+uint(d>0x80000u||(d==0x80000u&&(k&1u)));
 if(rr>>3){rr=0u;e+=1;} if(e>8)return (s<<4)|0x7eu; return (s<<4)|(uint(e+7)<<3)|rr;}
R3DEV uint sfp8v_bits(uint b){uint sv=(b&0x80u)<<24,e=(b>>3)&0xfu,m=b&7u;
 if(e==15u&&m==7u)return 0x7fc00000u;
 if(e==0u){union{float f;uint u;}z;z.f=float(m)*0.001953125f;return (z.u&0x7fffffffu)|sv;}
 return sv|(uint(e+120)<<23)|(m<<20);}
R3DEV float sfp8v(uint b){return val(sfp8v_bits(b));}
using f8v=float __attribute__((ext_vector_type(8)));
using i2v=int __attribute__((ext_vector_type(2)));
using h8v=_Float16 __attribute__((ext_vector_type(8)));
// fp8: per-wave 128 words (A rows stride 4 =64, B same) = 8KB total.
// f16: only used by single-wave FASTWAVE kernels -> one 512-word region (2KB).
__attribute__((shared)) uint slab8[16*128];
__attribute__((shared)) uint slab16[512];
__attribute__((shared)) uint fp8tab[257];
R3DEV void ensure_fp8tab(uint n){
 uint t=__builtin_amdgcn_workitem_id_x();
 for(uint b=t;b<256u;b+=n)fp8tab[b]=sfp8v_bits(b);
 asm volatile("s_waitcnt lgkmcnt(0)\ns_barrier":::"memory");
}
R3DEV float fp8val(uint b){union{uint u;float f;}t;t.u=fp8tab[b];return t.f;}
R3DEV uint cvt_pk_fp8(float x,float y,uint acc,bool hi){
 uint w=sq8(x)|(sq8(y)<<8); return hi?(acc&0x0000ffffu)|(w<<16):(acc&0xffff0000u)|w;}
R3DEV float cvt_f32_fp8(uint packed,uint word){return sfp8v((packed>>(8*word))&255u);}
// The MAC folds below must stay two-op (mul then add): LLVM otherwise emits
// v_fma_mix_f32 (f16-source FMA), whose single-rounding/denormal behavior
// differs from the gfx12 WMMA accumulation being modeled.
#pragma clang fp contract(off)
R3DEV f8v wmma_fp8(i2v a,i2v b,f8v c){
#ifdef DG_NOWMA
 #pragma unroll
 for(uint e=0;e<8;e++)c[e]+=float(uint(a.x^b.x));
 return c;
#else
 uint l=__builtin_amdgcn_workitem_id_x(),w=l>>5,ln=l&31,rc=ln&15,g=ln>>4;
 uint* A=slab8+w*128,*B=A+64;
 A[rc*4+g*2]=uint(a[0]); A[rc*4+g*2+1]=uint(a[1]);
 B[rc*4+g*2]=uint(b[0]); B[rc*4+g*2+1]=uint(b[1]);
 asm volatile("s_waitcnt lgkmcnt(0)":::"memory");
 f8v r;
 #pragma unroll
 for(uint e=0;e<8;e++){uint arow=(g*8+e)*4,brow=rc*4;
  uint av0=A[arow],av1=A[arow+1],av2=A[arow+2],av3=A[arow+3];
  uint bv0=B[brow],bv1=B[brow+1],bv2=B[brow+2],bv3=B[brow+3];
  float s=c[e];
  s=fp8val(av0&255u)*fp8val(bv0&255u)+s;
  s=fp8val((av0>>8)&255u)*fp8val((bv0>>8)&255u)+s;
  s=fp8val((av0>>16)&255u)*fp8val((bv0>>16)&255u)+s;
  s=fp8val(av0>>24)*fp8val(bv0>>24)+s;
  s=fp8val(av1&255u)*fp8val(bv1&255u)+s;
  s=fp8val((av1>>8)&255u)*fp8val((bv1>>8)&255u)+s;
  s=fp8val((av1>>16)&255u)*fp8val((bv1>>16)&255u)+s;
  s=fp8val(av1>>24)*fp8val(bv1>>24)+s;
  s=fp8val(av2&255u)*fp8val(bv2&255u)+s;
  s=fp8val((av2>>8)&255u)*fp8val((bv2>>8)&255u)+s;
  s=fp8val((av2>>16)&255u)*fp8val((bv2>>16)&255u)+s;
  s=fp8val(av2>>24)*fp8val(bv2>>24)+s;
  s=fp8val(av3&255u)*fp8val(bv3&255u)+s;
  s=fp8val((av3>>8)&255u)*fp8val((bv3>>8)&255u)+s;
  s=fp8val((av3>>16)&255u)*fp8val((bv3>>16)&255u)+s;
  s=fp8val(av3>>24)*fp8val(bv3>>24)+s;
  r[e]=s;}
 return r;
#endif
}
R3DEV f8v wmma_f16(h8v a,h8v b,f8v c){
 uint l=__builtin_amdgcn_workitem_id_x(),w=l>>5,ln=l&31,rc=ln&15,g=ln>>4;
 uint* A=slab16,*B=A+256; (void)w;
 #pragma unroll
 for(uint j=0;j<4;j++){A[rc*8+g*4+j]=hb16(a[2*j])|(hb16(a[2*j+1])<<16);
                       B[rc*8+g*4+j]=hb16(b[2*j])|(hb16(b[2*j+1])<<16);}
 asm volatile("s_waitcnt lgkmcnt(0)":::"memory");
 f8v r;
 #pragma unroll
 for(uint e=0;e<8;e++){uint arow=(g*8+e)*8,brow=rc*8;float s=c[e];
  #pragma unroll
  for(uint wd=0;wd<8;wd++){uint av=A[arow+wd],bv=B[brow+wd];
   #pragma unroll
   for(uint j=0;j<2;j++){
    union{uint u;unsigned short s[2];}ua,ub;ua.u=av;ub.u=bv;
    union{_Float16 h;unsigned short s;}ta{0},tb{0};ta.s=ua.s[j];tb.s=ub.s[j];
    s=s+float(ta.h)*float(tb.h);}}
  r[e]=s;}
 return r;}
R3DEV void bar(){asm volatile("s_waitcnt lgkmcnt(0)\ns_barrier":::"memory");}
}
