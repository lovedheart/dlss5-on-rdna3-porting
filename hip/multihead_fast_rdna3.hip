// gfx1103 (RDNA3 780M) port of multihead_fast.hip: WMMA + FP8 cvt emulated via
// rdna3_wmma_shim.h (must be concatenated before this file); the dense main loop
// is a register-blocked rewrite (2x4/thread, decoded-float LDS staging) proven
// bit-identical to the emulated-WMMA reference including the residual path.
// gfx1103 PORT: emulated WMMA via rdna3_wmma_shim.h (correctness-grade path).
#ifndef HIP_ISA_HALF
#define HIP_ISA_HALF 1
#endif
// Native gfx1201 FP8 WMMA multihead kernels. Original externalfloat32 ABI.
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
#ifndef HIP_NATIVE_FP8_F
#define HIP_NATIVE_FP8_F 1
#endif
#if HIP_NATIVE_FP8_F
DEV float F(float x){
 uint b=bits(x),a=b&0x7fffffffu;
 if(a==0)return 0.f;
 // Preserve the existing signed saturation, including nonfinite inputs.
 if(a>=0x43e00000u)return value((b&0x80000000u)|0x43e00000u);
 int packed=rdna3::cvt_pk_fp8(x,0.f,0,false);
 return rdna3::cvt_f32_fp8(packed,0);
}
#else
DEV float F(float x){uint b=bits(x),a=b&0x7fffffffu,s=b&0x80000000u;if(a==0)return 0.f;
 if(a>=0x43e00000u)return value(s|0x43e00000u); // finite E4M3 saturation, including overflow
 if(a<0x3c800000u){float scaled=absf(x)*512.f;uint q=uint(scaled);float r=scaled-float(q);q+=uint(r>.5f||(r==.5f&&(q&1u)));float f=float(q)/512.f;return s?-f:f;}
 uint r=(a+0x7ffffu+((a>>20)&1u))&0xfff00000u;return value(s|(r>0x43e00000u?0x43e00000u:r));}
#endif
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

// gfx1103 port: WMMA builtins replaced by rdna3:: emulation (see shim).
using i2=int __attribute__((ext_vector_type(2)));
using f8=float __attribute__((ext_vector_type(8)));
using f4=float __attribute__((ext_vector_type(4)));
#define TILE KERNEL __attribute__((amdgpu_flat_work_group_size(512,512)))
DEV uint q8(float x){return uint(rdna3::cvt_pk_fp8(x,x,0,false))&255u;}
DEV void pack(i2&v,uint e,float x){v[e/4]=int(uint(v[e/4])|(q8(x)<<(8*(e%4))));}
// Diagnostic production HLSL f32tof16 path: explicit RTZ, distinct from
// Cast<F16> / ISA_HALF RNE. Keep both exported contracts until measured.
DEV float Hrtz(float x){uint h;float out;asm("v_cvt_pkrtz_f16_f32 %0, %1, %1" : "=v"(h) : "v"(x));asm("v_cvt_f32_f16 %0, %1" : "=v"(out) : "v"(h));return out;}
//64x64 output tile,16 waves. Each thread stages one packed4-byte A row word
// and one B column word; each packed operand is reused by four waves.
DEV uint pack4(float a,float b,float c,float d){uint lo=uint(rdna3::cvt_pk_fp8(a,b,0,false));uint hi=uint(rdna3::cvt_pk_fp8(c,d,0,false));return (lo&65535u)|((hi&65535u)<<16);}
DEV float fp8d(float x){return rdna3::sfp8v(rdna3::sq8(x));}
DEV void fast_dense(const float* input,const float* feature,const float* weights,float* output,
 uint M,uint N,uint K,uint matrix_offset,uint skip_offset,uint epilogue,uint width=0,uint valid_width=0,uint valid_height=0,uint matrix_residual=0){
 // Register-blocked RDNA3 main loop (1103b treatment). Staging holds the SAME
 // fp8-quantised values decoded once to f32; each thread folds a 2x4 output
 // tile. Products fold into the live accumulator in ascending k within each
 // 16-K half and the two halves run in order, reproducing the two-WMMA-op
 // fold ((...((c+p0)+p1)...+p15)+q0...+q31) bit-exactly.
 __attribute__((shared)) float sm[4096];        // sa = [8 words][64 rows] float4, sb = [32 k][64 cols]
 float* sa=sm; float* sb=sm+2048;
 uint block=__builtin_amdgcn_workgroup_id_x(),nt=(N+63)/64;
 if(!nt||block>=((M+63)/64)*nt)return;
 uint mr=block/nt*64,nc=block%nt*64,t=__builtin_amdgcn_workitem_id_x();
 rdna3::ensure_fp8tab(512u);
 uint tr=t/16,tc=t%16;
 float c[2][4];
 #pragma unroll
 for(uint i=0;i<2;i++)
  #pragma unroll
  for(uint j=0;j<4;j++){uint p=mr+tr*2+i,n=nc+tc*4+j;c[i][j]=(p<M&&n<N&&skip_offset!=~0u&&!matrix_residual)?Hrtz(feature[p*N+n]*weights[skip_offset+n]):0.f;}
 if(matrix_residual){
  // Exact fragment arithmetic of the WMMA residual path, then a bit-preserving
  // LDS transfer into the register-tile accumulators.
  uint wave=t/32,l=t%32,rc=l&15,group=l>>4,wm=wave/4,wn=wave%4;
  uint first=mr+wm*16,col=nc+wn*16;
  f8 accum{};
  uint n=col+rc;float remain=n<N?weights[skip_offset+n]:0.f;
  for(uint part=0;part<3;part++){
   float component;
   if(absf(remain)<.015625f){float x=absf(remain)*512.f;uint q=uint(x);float r=x-float(q);q+=uint(r>.5f||(r==.5f&&(q&1)));q=q>7?7:q;component=(remain<0?-1.f:1.f)*float(q)/512.f;}
   else component=F(remain);
   remain-=component;
   #pragma unroll
   for(uint half=0;half<2;half++){i2 a{},b{};
    #pragma unroll
    for(uint e=0;e<8;e++){uint k=half*16+group*8+e,p=first+rc,g=col/32*32;float x=(p<M&&g+k<N)?feature[p*N+g+k]:0.f;pack(a,e,x);pack(b,e,(n<N&&k==n%32)?component:0.f);}
    accum=rdna3::wmma_fp8(a,b,accum);
   }
  }
  #pragma unroll
  for(uint e=0;e<8;e++)sm[(first+group*8+e-mr)*64+col+rc-nc]=accum[e];
  rdna3::bar();
  #pragma unroll
  for(uint i=0;i<2;i++)
   #pragma unroll
   for(uint j=0;j<4;j++)c[i][j]=sm[(tr*2+i)*64+tc*4+j];
 }
 for(uint chunk=0;chunk<K/32;chunk++){
  uint row=t/8,word=t%8,k=chunk*32+word*4,p=mr+row,n=nc+row;
  float a0=0,a1=0,a2=0,a3=0,b0=0,b1=0,b2=0,b3=0;
  if(p<M&&(!valid_width||(p%width<valid_width&&p/width<valid_height))){uint off=p*K+k;a0=input[off];a1=input[off+1];a2=input[off+2];a3=input[off+3];}
  #ifndef HIP_PREPACKED_WEIGHTS
  if(n<N){uint off=matrix_offset+n*K+k;b0=weights[off];b1=weights[off+1];b2=weights[off+2];b3=weights[off+3];}
#endif
  {f4 v={fp8d(a0),fp8d(a1),fp8d(a2),fp8d(a3)};*(f4*)(sa+(word*64+row)*4)=v;}
#ifdef HIP_PREPACKED_WEIGHTS
  // Matrix bytes start at the original float-region offset; nonmatrix fields stay f32.
  const uint* packed_weights=reinterpret_cast<const uint*>(weights);
  {uint w=n<N?packed_weights[matrix_offset+(n*K+k)/4]:0u;
   #pragma unroll
   for(uint j=0;j<4;j++)sb[(word*4+j)*64+row]=rdna3::sfp8v((w>>(8*j))&255u);}
#else
  uint kw=(word*4)*64+row;sb[kw]=fp8d(b0);sb[kw+64]=fp8d(b1);sb[kw+128]=fp8d(b2);sb[kw+192]=fp8d(b3);
#endif
  rdna3::bar();
  #pragma unroll
  for(uint half=0;half<2;half++){
   #pragma unroll
   for(uint kk=0;kk<16;kk++){uint k2=half*16+kk,word2=k2>>2,j2=k2&3;
    float ra0=sa[(word2*64+tr*2)*4+j2],ra1=sa[(word2*64+tr*2+1)*4+j2];
    f4 rb=*(const f4*)(sb+k2*64+tc*4);
    c[0][0]=c[0][0]+ra0*rb.x;c[0][1]=c[0][1]+ra0*rb.y;c[0][2]=c[0][2]+ra0*rb.z;c[0][3]=c[0][3]+ra0*rb.w;
    c[1][0]=c[1][0]+ra1*rb.x;c[1][1]=c[1][1]+ra1*rb.y;c[1][2]=c[1][2]+ra1*rb.z;c[1][3]=c[1][3]+ra1*rb.w;
   }
  }
  // All threads finish reading before any writer reuses this K32 tile.
  rdna3::bar();
 }
 #pragma unroll
 for(uint i=0;i<2;i++)
  #pragma unroll
  for(uint j=0;j<4;j++){uint p=mr+tr*2+i,n=nc+tc*4+j;if(p>=M||n>=N)continue;float a=c[i][j],v;
  if(epilogue==1){float gate=clampf(a,-4.f,4.f),q=__builtin_fmaf(absf(gate),-.055908203125f,.447265625f),poly=__builtin_fmaf(gate,q,.89453125f);v=F(a*poly);}
  else v=epilogue==2?a:epilogue==3?Hrtz(a):epilogue==4?F(a):epilogue==5?F(Hrtz(a)):F(H(a));
  output[p*N+n]=(valid_width&&(p%width>=valid_width||p/width>=valid_height))?0.f:v;
 }
}
// Final build contract: fused FP8act overwritten without HW_QUANTIZE:
// expand=F(polynomial(FP32 acc)), contract=F(H(FP32 acc)).
TILE void mh_ffn_expand_fast(const float* input,const float* weights,float* hidden,uint tokens,uint channels){if(valid_c(channels))fast_dense(input,input,weights,hidden,tokens,4*channels,channels,0,~0u,1);}
TILE void mh_ffn_contract_fast(const float* hidden,const float* weights,float* middle,uint tokens,uint channels){if(valid_c(channels))fast_dense(hidden,hidden,weights,middle,tokens,channels,4*channels,4*channels*channels,~0u,5);}
// Production MAP_FEATURE+DIRECT_CAST variant. Not MATRIX_RESIDUAL.
TILE void mh_ffn_project_fast(const float* middle,const float* feature,const float* weights,float* output,uint tokens,uint channels){if(valid_c(channels))fast_dense(middle,feature,weights,output,tokens,channels,channels,8*channels*channels,9*channels*channels,4);}
// Raw FP32 QKV; do NOT feed to legacy H-reduction normalization.
TILE void mh_qkv_fast(const float* input,const float* weights,float* qkv,uint tokens,uint channels){if(valid_attn(channels))fast_dense(input,input,weights,qkv,tokens,3*channels,channels,0,~0u,2);}
// Explicit scalar-residual diagnostic variant. Does NOT implement production
// NATIVE_MATRIX_RESIDUAL (three FP8 diagonal MMAs). post:0 F(H),3 H,4 directF.
TILE void mh_attention_project_fast_scalar(const float* av,const float* feature,const float* weights,float* output,uint tokens,uint channels,uint post){if(valid_attn(channels)&&(post==0||post==3||post==4))fast_dense(av,feature,weights,output,tokens,channels,channels,3*channels*channels,4*channels*channels+(channels/32)*4096+channels/32,post==0?5:post);}

// Same input/weight/output ABI as contract_fast, only the last epilogue differs.
TILE void mh_ffn_contract_fast_pre(const float* hidden,const float* weights,float* middle,uint tokens,uint channels){if(valid_c(channels))fast_dense(hidden,hidden,weights,middle,tokens,channels,4*channels,4*channels*channels,~0u,2);}
TILE void mh_ffn_contract_fast_rtz(const float* hidden,const float* weights,float* middle,uint tokens,uint channels){if(valid_c(channels))fast_dense(hidden,hidden,weights,middle,tokens,channels,4*channels,4*channels*channels,~0u,5);}

TILE void mh_ffn_contract_fast_rne(const float* hidden,const float* weights,float* middle,uint tokens,uint channels){if(valid_c(channels))fast_dense(hidden,hidden,weights,middle,tokens,channels,4*channels,4*channels*channels,~0u,0);}
// Production NATIVE_QKV_FAST2=0: rawFP32 squares, sequentialFP32 sum,
// unrounded rsqrt*scale and direct saturated FP8 cast. Not legacy normalize.
// Input/output HWC with three parts[p][Q,K,V][C]. Block256, one thread per head/token.
KERNEL __attribute__((amdgpu_flat_work_group_size(256,256)))
void mh_qkv_normalize_fast(const float* qkv,const float* weights,float* normalized,uint tokens,uint channels){
 uint i=lane(),heads=channels/32;if(!valid_attn(channels)||i>=tokens*heads)return;uint p=i/heads,head=i%heads,scale=4*channels*channels+heads*4096;
 for(uint part=0;part<3;part++){uint base=(p*3+part)*channels+head*32;float inv=1.f;
  if(part<2){float ss=0.f;for(uint c=0;c<32;c++){float v=qkv[base+c];float square=v*v;ss+=square;}inv=__builtin_amdgcn_rsqf(maxf(ss,6.198883056640625e-5f))*(part==0?weights[scale+head]:1.f);}
  for(uint c=0;c<32;c++)normalized[base+c]=F(qkv[base+c]*inv);
 }
}

// Matrix-residual projection: three original FP8 diagonal terms, no seed H.
// post0 F(H_RTZ),3 H_RTZ,4 directFP8; feature must be exact finiteFP8.
TILE void mh_attention_project_fast_matrix(const float* av,const float* feature,const float* weights,float* output,uint tokens,uint channels,uint post){if(valid_attn(channels)&&(post==0||post==3||post==4))fast_dense(av,feature,weights,output,tokens,channels,channels,3*channels*channels,4*channels*channels+(channels/32)*4096+channels/32,post==0?5:post,0,0,0,1);}
#define FASTWAVE KERNEL __attribute__((amdgpu_flat_work_group_size(32,32)))
using h8=_Float16 __attribute__((ext_vector_type(8)));
// normalized is raster[p][Q,K,V][C], ex/prob are[window][head][q64][k64].
FASTWAVE void mh_scores_exp_fast(const float* normalized,const float* weights,float* ex,uint width,uint height,uint channels){rdna3::ensure_fp8tab(32u);
 uint tile=__builtin_amdgcn_workgroup_id_x(),heads=channels/32,windows=width/8*(height/8);if(!valid_attn(channels)||width%8||height%8||tile>=windows*heads*16)return;
 uint wh=tile/16,win=wh/heads,head=wh%heads,q=tile%16/4*16,col=tile%4*16,l=__builtin_amdgcn_workitem_id_x(),rc=l&15,g=l>>4;f8 sum{};
 for(uint half=0;half<2;half++){i2 a{},b{};for(uint e=0;e<8;e++){uint c=half*16+g*8+e,qp=pixel_in_window(win,q+rc,width),kp=pixel_in_window(win,col+rc,width);pack(a,e,normalized[(qp*3)*channels+head*32+c]);pack(b,e,normalized[(kp*3+1)*channels+head*32+c]);}sum=rdna3::wmma_fp8(a,b,sum);}
 for(uint e=0;e<8;e++){uint qr=q+g*8+e,k=col+rc;float score=sum[e]+weights[4*channels*channels+head*4096+qr*64+k];float a=clampf(score*.044921875f+1.30078125f,1.03125f,1.5693359375f);uint h;asm("v_cvt_pkrtz_f16_f32 %0, %1, %1" : "=v"(h) : "v"(a));ex[(wh*64+qr)*64+k]=from_half((((h&65535u)<<5)+0x8000u)&65535u);}
}
// Each wave handles16 queries, computes the two production32-key partials
// with F16 WMMA against ones, then forms probabilities with unrounded inverse.
FASTWAVE void mh_probabilities_fast(const float* ex,float* prob,uint windows,uint channels){
 uint tile=__builtin_amdgcn_workgroup_id_x(),heads=channels/32;if(!valid_attn(channels)||tile>=windows*heads*4)return;uint wh=tile/4,q=tile%4*16,l=__builtin_amdgcn_workitem_id_x(),rc=l&15,g=l>>4;f8 totals[2];
 for(uint side=0;side<2;side++){f8 sum{};for(uint half=0;half<2;half++){h8 a{},b{};for(uint e=0;e<8;e++){uint k=side*16+half*32+g*8+e;a[e]=(_Float16)ex[(wh*64+q+rc)*64+k];b[e]=(_Float16)1.f;}sum=rdna3::wmma_f16(a,b,sum);}totals[side]=sum;}
 for(uint e=0;e<8;e++){uint qr=q+g*8+e;float inv=1.f/(totals[0][e]+totals[1][e]);for(uint k=rc;k<64;k+=16)prob[(wh*64+qr)*64+k]=F(ex[(wh*64+qr)*64+k]*inv);}
}
FASTWAVE void mh_attention_av_fast(const float* prob,const float* normalized,float* av,uint width,uint height,uint channels){rdna3::ensure_fp8tab(32u);
 uint tile=__builtin_amdgcn_workgroup_id_x(),heads=channels/32,windows=width/8*(height/8);if(!valid_attn(channels)||width%8||height%8||tile>=windows*heads*8)return;
 uint wh=tile/8,win=wh/heads,head=wh%heads,q=tile%8/2*16,col=tile%2*16,l=__builtin_amdgcn_workitem_id_x(),rc=l&15,g=l>>4;f8 sum{};
 for(uint half=0;half<4;half++){i2 a{},b{};for(uint e=0;e<8;e++){uint k=half*16+g*8+e,kp=pixel_in_window(win,k,width);pack(a,e,prob[(wh*64+q+rc)*64+k]);pack(b,e,normalized[(kp*3+2)*channels+head*32+col+rc]);}sum=rdna3::wmma_fp8(a,b,sum);}
 for(uint e=0;e<8;e++){uint p=pixel_in_window(win,q+g*8+e,width);av[p*channels+head*32+col+rc]=F(sum[e]);}
}
// Production pool projection is NOT the dense fast accumulation schedule.
// C>=64: F16 operands, independent K32 sum then software-H/RNE each group.
// C32: FP8 operands, a single K32 dot, explicit HLSL-half/RTZ epilogue.
FASTWAVE void mh_pool_project_production(const float* pooled,const float* weights,float* output,uint width,uint height,uint valid_width,uint valid_height,uint channels){
 uint M=width*height,N=2*channels,tile=__builtin_amdgcn_workgroup_id_x(),nt=N/16;
 if(!valid_spatial(channels)||tile>=((M+15)/16)*nt)return;
 if(channels==32)rdna3::ensure_fp8tab(32u); // emulated wmma_fp8 decode table
 uint first=tile/nt*16,col=tile%nt*16,l=__builtin_amdgcn_workitem_id_x(),rc=l&15,g=l>>4;f8 acc{};
 for(uint chunk=0;chunk<channels/32;chunk++){f8 sum{};
  for(uint half=0;half<2;half++){
   if(channels==32){i2 a{},b{};for(uint e=0;e<8;e++){uint k=half*16+g*8+e,p=first+rc;pack(a,e,p<M?pooled[p*channels+k]:0.f);pack(b,e,weights[(col+rc)*channels+k]);}sum=rdna3::wmma_fp8(a,b,sum);}
   else{h8 a{},b{};for(uint e=0;e<8;e++){uint k=chunk*32+half*16+g*8+e,p=first+rc;a[e]=(_Float16)(p<M?pooled[p*channels+k]:0.f);b[e]=(_Float16)weights[(col+rc)*channels+k];}sum=rdna3::wmma_f16(a,b,sum);}
  }
  for(uint e=0;e<8;e++)acc[e]=channels==32?Hrtz(acc[e]+sum[e]):H(acc[e]+sum[e]);
 }
 for(uint e=0;e<8;e++){uint p=first+g*8+e;if(p<M)output[p*N+col+rc]=(valid_width&&(p%width>=valid_width||p/width>=valid_height))?0.f:F(acc[e]);}
}
