// gfx1103 (RDNA3 780M) port of multihead_tiled.hip: software FP8 E4M3 + LDS-staged
// register-blocked 4x4 tiles replacing the gfx12 WMMA ops. Bit-exact vs the gfx12
// family. __syncthreads must be an explicit lgkmcnt barrier on RDNA3.
#define __syncthreads() asm volatile("s_waitcnt lgkmcnt(0)\ns_barrier" ::: "memory")
// gfx1103 PORT of the multihead tiled family. Bit-faithful except WMMA's
// internal 16-term accumulation order (see NOTES.md phase 3).
// Mirrors native_c64.hlsl non-wave math, native_c64_shift.hlsl, native_c64_ds.hlsl.
// Host validates geometry/channel contracts; all tensors/weights externally float32.
// Every exported kernel uses one-dimensional blocks of exactly512 threads.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
typedef unsigned uint;
#define DEV __attribute__((device)) __attribute__((always_inline))
#define KERNEL extern "C" __attribute__((global))
DEV uint bits(float x){union{float f;uint u;}v;v.f=x;return v.u;}
DEV float value(uint x){union{float f;uint u;}v;v.u=x;return v.f;}
DEV uint lane(){return __builtin_amdgcn_workgroup_id_x()*256u+__builtin_amdgcn_workitem_id_x();}
DEV float absf(float x){return value(bits(x)&0x7fffffffu);}
DEV float maxf(float a,float b){return a>b?a:b;}
DEV float minf(float a,float b){return a<b?a:b;}
DEV float clampf(float a,float lo,float hi){return minf(maxf(a,lo),hi);}
DEV uint shift_rne(uint x,uint n){if(n>31)return 0;uint q=x>>n,mask=(1u<<n)-1u,r=x&mask,mid=1u<<(n-1);return q+uint(r>mid||(r==mid&&(q&1u)));}
DEV uint half_bits(float x){uint b=bits(x),a=b&0x7fffffffu,s=(b>>16)&0x8000u;int e=int(a>>23)-127;
 if(a>=0x7f800000u)return s|0x7c00u|((a&0x7fffffu)?0x200u:0u);
 if(e<-25)return s;
 if(e<-14)return s|shift_rne((a&0x7fffffu)|0x800000u,uint(-e-1));
 if(e>15)return s|0x7c00u;
 uint h=((a+0xfffu+((a>>13)&1u))>>13)-0x1c000u;return s|(h>=0x7c00u?0x7c00u:h);
}
DEV float from_half(uint h){uint s=(h&0x8000u)<<16,e=(h>>10)&31u,m=h&1023u;if(e==31)return value(s|0x7f800000u|(m<<13));if(!e){if(!m)return value(s);float f=float(m)*5.9604644775390625e-8f;return s?-f:f;}return value(s|((e+112)<<23)|(m<<13));}
#ifdef HIP_ISA_HALF
// Keep f32 expression evaluation and both rounding boundaries opaque to fusion.
DEV float H(float x){uint h;float result;asm("v_cvt_f16_f32 %0, %1" : "=v"(h) : "v"(x));asm("v_cvt_f32_f16 %0, %1" : "=v"(result) : "v"(h));return result;}
#else
DEV float H(float x){if((bits(x)&0x7fffffffu)>=0x7f800000u)return x;return from_half(half_bits(x));}
#endif
DEV float F(float x){uint b=bits(x),a=b&0x7fffffffu,s=b&0x80000000u;if(a==0)return 0.f;
 if(a>=0x43e00000u)return value(s|0x43e00000u); // finite E4M3 saturation, including overflow
 if(a<0x3c800000u){float scaled=absf(x)*512.f;uint q=uint(scaled);float r=scaled-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));float f=float(q)/512.f;return s?-f:f;}
 uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;return value(s|(r>0x43e00000u?0x43e00000u:r));}
DEV float half_square_pair(float lower,float upper){float a=lower*lower,b=H(upper*upper),sum=a+b;
 if((bits(sum)&0x7fffffffu)>=0x7f800000u)return H(sum);
 float large=maxf(a,b),small=minf(a,b),recovered=sum-large,error=small-recovered;
 if(sum==65520.f)return error<0?65504.f:H(sum);
 float rounded=H(sum);if(error==0||(bits(rounded)&0x7fffffffu)>=0x7f800000u||rounded==sum)return rounded;
 uint hb=half_bits(rounded),lo_bits=sum<rounded?hb-1:hb;float lo=from_half(lo_bits),hi=from_half(lo_bits+1),mid=(lo+hi)*.5f;
 return sum==mid?(error>0?hi:lo):rounded;
}
DEV float exp_approx(float x){float a=clampf(H(x*.044921875f+1.30078125f),1.03125f,1.5693359375f);return from_half(((half_bits(a)<<5)+0x8000u)&65535u);}
DEV bool valid_c(uint c){return c==64u||c==128u||c==256u;}
DEV bool valid_attn(uint c){return valid_c(c)||c==512u;}
DEV bool valid_spatial(uint c){return c==32u||valid_attn(c);}
DEV uint pixel_in_window(uint window,uint token,uint width){return ((window/(width/8))*8+token/8)*width+(window%(width/8))*8+token%8;}
DEV uint window_of_pixel(uint p,uint width){return (p/width/8)*(width/8)+(p%width)/8;}
DEV uint token_of_pixel(uint p,uint width){return ((p/width)%8)*8+(p%width)%8;}

using i2=int __attribute__((ext_vector_type(2)));
using f8=float __attribute__((ext_vector_type(8)));
#define TILE KERNEL __attribute__((amdgpu_flat_work_group_size(256,256)))
DEV float fp8v(uint b){uint s=(b&0x80u)<<24,e=(b>>3)&0xfu,m=b&7u;if(e==15u&&m==7u)return value(0x7fc00000u);if(e==0u)return s?-float(m)*0.001953125f:float(m)*0.001953125f;return value(s|(uint(e+120)<<23)|(m<<20));}
DEV uint q8(float x){uint b=bits(x),a=b&0x7fffffffu,s=(b>>28)&8u;
 if(a>=0x7f800000u)return (s<<4)|0x7fu;
 if(a>=0x43e00000u)return (s<<4)|0x7eu;
 if(a<0x3c800000u){float sc=absf(x)*512.f;uint q=uint(sc),r=bits(sc-float(q));q+=uint(r>0x3f000000u||(r==0x3f000000u&&(q&1u)));return (s<<4)|(q>7u?8u:q);}
 int e=int(a>>23)-127;uint m=a&0x7fffffu,keep=m>>20,drop=m&0xfffffu;
 uint r=keep+uint(drop>0x80000u||(drop==0x80000u&&(keep&1u)));
 if(r>>3){r=0;e+=1;} if(e>8)return (s<<4)|0x7eu; return (s<<4)|(uint(e+7)<<3)|r;}
DEV void pack(i2&v,uint e,float x){v[e/4]=int(uint(v[e/4])|(q8(x)<<(8*(e%4))));}
//64x64 output tile,16 waves. Each thread stages one packed4-byte A row word
// and one B column word; each packed operand is reused by four waves.
DEV uint pack4(float a,float b,float c,float d){return q8(a)|(q8(b)<<8)|(q8(c)<<16)|(q8(d)<<24);}
DEV void tiled_dense(const float* input,const float* feature,const float* weights,float* output,
 uint M,uint N,uint K,uint matrix_offset,uint skip_offset,uint epilogue,uint width=0,uint valid_width=0,uint valid_height=0){
 // Register-blocked port: stage the fp8-quantised values already decoded to f32
 // (q8->fp8v per staged value = exactly the byte the WMMA path would have used,
 // decoded once instead of 8x per fragment read). 256 threads, 4x4 register
 // tile, K32 chunks. Summation per C element stays word-major wd0..3, byte j
 // 0..3, so results are bit-identical to the phase-3 LDS port and its mirror.
 __attribute__((shared)) float sa[32][68];   // [k][row]  (pad 4; staging writes 64..67 too)
 __attribute__((shared)) float sb[68][32];   // [col][k]
 uint block=__builtin_amdgcn_workgroup_id_x(),nt=(N+63)/64;
 if(!nt||block>=((M+63)/64)*nt)return;
 uint mr=block/nt*64,nc=block%nt*64,t=__builtin_amdgcn_workitem_id_x();
 uint tr=t/16,tc=t%16;
 float c[4][4];
 #pragma unroll
 for(uint i=0;i<4;i++)
  #pragma unroll
  for(uint j=0;j<4;j++){uint p=mr+tr*4+i,n=nc+tc*4+j;c[i][j]=(p<M&&n<N&&skip_offset!=~0u)?H(feature[p*N+n]*weights[skip_offset+n]):0.f;}
 for(uint chunk=0;chunk<K/32;chunk++){
  uint row=t/4,sub=t%4,p=mr+row,n=nc+row;
  #pragma unroll
  for(uint h=0;h<2;h++){
   uint kb=chunk*32+sub*4+h*16;
   #pragma unroll
   for(uint j=0;j<4;j++){
    float v=((p<M)&&(!width||(p%width<valid_width&&p/width<valid_height)))?input[p*K+kb+j]:0.f;
    sa[sub*4+h*16+j][row]=fp8v(q8(v));}
   #pragma unroll
   for(uint j=0;j<4;j++){float v=(n<N)?weights[matrix_offset+n*K+kb+j]:0.f;sb[row][sub*4+h*16+j]=fp8v(q8(v));}
  }
  __syncthreads();
  float st[4][4];
  #pragma unroll
  for(uint i=0;i<4;i++)
   #pragma unroll
   for(uint j=0;j<4;j++)st[i][j]=0.f;
  #pragma unroll
  for(uint k=0;k<32;k++){
   float ra[4],rb[4];
   #pragma unroll
   for(uint i=0;i<4;i++)ra[i]=sa[k][tr*4+i];
   #pragma unroll
   for(uint j=0;j<4;j++)rb[j]=sb[tc*4+j][k];
   #pragma unroll
   for(uint i=0;i<4;i++)
    #pragma unroll
    for(uint j=0;j<4;j++)st[i][j]+=ra[i]*rb[j];
  }
  #pragma unroll
  for(uint i=0;i<4;i++)
   #pragma unroll
   for(uint j=0;j<4;j++)c[i][j]=H(c[i][j]+st[i][j]);   // the WMMA path rounds at every K32
  __syncthreads();
 }
 #pragma unroll
 for(uint i=0;i<4;i++)
  #pragma unroll
  for(uint j=0;j<4;j++){uint p=mr+tr*4+i,n=nc+tc*4+j;if(p>=M||n>=N)continue;float a=c[i][j],v;
   if(epilogue==1){float gate=clampf(a,-4.f,4.f),poly=H(gate*H(absf(gate)*(-.055908203125f)+.447265625f)+.89453125f);v=F(H(a*poly));}
   else v=epilogue==2?a:F(a);
   output[p*N+n]=(width&&(p%width>=valid_width||p/width>=valid_height))?0.f:v;}
}
TILE void mh_ffn_expand_tiled(const float* input,const float* weights,float* hidden,uint tokens,uint channels){if(valid_c(channels))tiled_dense(input,input,weights,hidden,tokens,4*channels,channels,0,~0u,1);}
TILE void mh_ffn_contract_tiled(const float* hidden,const float* weights,float* middle,uint tokens,uint channels){if(valid_c(channels))tiled_dense(hidden,hidden,weights,middle,tokens,channels,4*channels,4*channels*channels,~0u,0);}
TILE void mh_ffn_project_tiled(const float* middle,const float* feature,const float* weights,float* output,uint tokens,uint channels){if(valid_c(channels))tiled_dense(middle,feature,weights,output,tokens,channels,channels,8*channels*channels,9*channels*channels,0);}
TILE void mh_qkv_tiled(const float* input,const float* weights,float* qkv,uint tokens,uint channels){if(valid_attn(channels))tiled_dense(input,input,weights,qkv,tokens,3*channels,channels,0,~0u,2);}
TILE void mh_attention_project_tiled(const float* av,const float* feature,const float* weights,float* output,uint tokens,uint channels,uint raw_output){if(valid_attn(channels))tiled_dense(av,feature,weights,output,tokens,channels,channels,3*channels*channels,4*channels*channels+(channels/32)*4096+channels/32,raw_output?2:0);}
TILE void mh_pool_project_tiled(const float* pooled,const float* weights,float* output,uint width,uint height,uint valid_width,uint valid_height,uint channels){if(valid_spatial(channels))tiled_dense(pooled,pooled,weights,output,width*height,2*channels,channels,0,~0u,0,width,valid_width,valid_height);}
// Arbitrary dense sizes for tail-N32/M400 and residual-placement validation.
// This extra entry is test-only; original family exports above retain their ABI.
TILE void mh_dense_tiled_probe(const float* input,const float* feature,const float* weights,float* output,uint M,uint N,uint K,uint skip_offset,uint raw){if(K&&!(K%32))tiled_dense(input,feature,weights,output,M,N,K,0,skip_offset,raw?2:0);}
