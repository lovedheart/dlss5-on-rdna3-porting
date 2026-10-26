#!/bin/bash
# e2e build of gfx1103 module .hsaco files, replicating the gfx1103 rows of
# patchtree hip/build-modules.ps1 (defines prepended, sources concatenated) with
# the 3-stage recipe from Development/tools/llvm-fork/compile-modules.py:
#   clang --offload-device-only -emit-llvm -> clang -target amdgcn (internalize) -> ld.lld --no-undefined -shared
set -u
E2E=$(cd "$(dirname "$0")" && pwd)
CLANG=/opt/rocm/lib/llvm/bin/clang
LLD=/opt/rocm/lib/llvm/bin/ld.lld
REPO=/home/lovedheart/dlss5-on-amd-9070xt-porting
PT=/home/lovedheart/.copilot/session-state/ce73c91d-9e5d-444f-ae79-54d9fe8aded9/files/gfx1103-proto/phase5/patchtree/hip
PROTO=/home/lovedheart/.copilot/session-state/ce73c91d-9e5d-444f-ae79-54d9fe8aded9/files/gfx1103-proto
GEN=$E2E/modules/gen
OUT=$E2E/modules
mkdir -p "$GEN" "$OUT"

build_one() { # name, defines..., --, source files...
  local name=$1; shift
  local defines=(); while [[ $1 != -- ]]; do defines+=("$1"); shift; done; shift
  local g=$GEN/$name.generated.hip
  { printf '#define HIP_ISA_HALF 1\n'; for d in "${defines[@]}"; do printf '#define %s\n' "$d"; done
    # COMGR device-only mode has no hip headers; strip the host-side include and supply the
    # same lgkmcnt barrier #define the patchtree ports use (bit-identical, barrier only)
    printf '#ifndef __syncthreads\n#define __syncthreads() asm volatile("s_waitcnt lgkmcnt(0)\\ns_barrier" ::: "memory")\n#endif\n'
    for s in "$@"; do sed '/#include <hip\/hip_runtime\.h>/d' "$s"; printf '\n'; done; } > "$g"
  local bc=$GEN/$name.bc obj=$GEN/$name.o h=$OUT/$name.hsaco
  "$CLANG" --target=x86_64-pc-linux-gnu --offload-arch=gfx1103 -O3 -x hip --offload-device-only \
      -cuid=0 -c -emit-llvm -nogpuinc -nogpulib -std=c++14 "$g" -o "$bc" > "$GEN/$name.log" 2>&1 \
    || { echo "FAIL(front) $name"; tail -5 "$GEN/$name.log"; return 1; }
  "$CLANG" -target amdgcn-amd-amdhsa -mcpu=gfx1103 -O3 -nogpulib -c -mllvm -amdgpu-internalize-symbols \
      "$bc" -o "$obj" >> "$GEN/$name.log" 2>&1 \
    || { echo "FAIL(backend) $name"; tail -5 "$GEN/$name.log"; return 1; }
  "$LLD" --no-undefined -shared -plugin-opt=mcpu=gfx1103 "$obj" -o "$h" >> "$GEN/$name.log" 2>&1 \
    || { echo "FAIL(link) $name"; tail -5 "$GEN/$name.log"; return 1; }
  echo "built $name $(stat -c%s "$h") bytes"
}

# boundary_fast.hip: gfx12 fp8-cvt builtin -> shim replacement (same sed family as the *_rdna3 ports)
sed 's/__builtin_amdgcn_cvt_f32_fp8(/rdna3::cvt_f32_fp8(/g' "$REPO/hip/boundary_fast.hip" > "$GEN/boundary_fast_rdna3.hip"

R=$REPO/hip
p () { echo "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8"; }

jobs=()
case "${1:-all}" in
 c32_prefix_reference) build_one c32_prefix_reference -- "$R/c32_reference.hip" "$R/prefix_reference.hip";;
 multihead-reference)  build_one multihead-reference  -- "$R/multihead_reference.hip";;
 deep_reference)       build_one deep_reference       -- "$R/deep_reference.hip";;
 boundary_reference)   build_one boundary_reference   -- "$R/boundary_reference.hip";;
 multihead-tiled)      build_one multihead-tiled      -- "$PROTO/multihead_tiled_1103.hip";;
 c32_tiled)            build_one c32_tiled            -- "$PROTO/c32_tiled_1103.hip";;
 multihead-fast)       build_one multihead-fast       -- "$PT/rdna3_wmma_shim.h" "$PT/multihead_fast_rdna3.hip";;
 multihead-fast-packed) build_one multihead-fast-packed "HIP_PREPACKED_WEIGHTS 1" -- "$PT/rdna3_wmma_shim.h" "$PT/multihead_fast_rdna3.hip";;
 prefix_fast)          build_one prefix_fast          -- "$PT/rdna3_wmma_shim.h" "$PT/prefix_fast_rdna3.hip";;
 c32_fast)             build_one c32_fast             -- "$PT/rdna3_wmma_shim.h" "$PT/c32_fast_rdna3.hip";;
 c32_fast_attention)   build_one c32_fast_attention   -- "$PT/rdna3_wmma_shim.h" "$PT/c32_fast_attention_rdna3.hip";;
 boundary-fast)        build_one boundary-fast        -- "$PT/rdna3_wmma_shim.h" "$PT/c32_fast_attention_rdna3.hip" "$GEN/boundary_fast_rdna3.hip";;
 deep_fast)            build_one deep_fast            -- "$PT/rdna3_wmma_shim.h" "$PT/deep_fast_rdna3.hip";;
 all)
   build_one c32_prefix_reference -- "$R/c32_reference.hip" "$R/prefix_reference.hip" &
   build_one multihead-reference  -- "$R/multihead_reference.hip" &
   build_one deep_reference       -- "$R/deep_reference.hip" &
   build_one boundary_reference   -- "$R/boundary_reference.hip" &
   build_one multihead-tiled      -- "$PROTO/multihead_tiled_1103.hip" &
   build_one c32_tiled            -- "$PROTO/c32_tiled_1103.hip" &
   wait
   build_one multihead-fast       -- "$PT/rdna3_wmma_shim.h" "$PT/multihead_fast_rdna3.hip" &
   build_one prefix_fast          -- "$PT/rdna3_wmma_shim.h" "$PT/prefix_fast_rdna3.hip" &
   build_one c32_fast             -- "$PT/rdna3_wmma_shim.h" "$PT/c32_fast_rdna3.hip" &
   build_one c32_fast_attention   -- "$PT/rdna3_wmma_shim.h" "$PT/c32_fast_attention_rdna3.hip" &
   build_one boundary-fast        -- "$PT/rdna3_wmma_shim.h" "$PT/c32_fast_attention_rdna3.hip" "$GEN/boundary_fast_rdna3.hip" &
   build_one multihead-fast-packed "HIP_PREPACKED_WEIGHTS 1" -- "$PT/rdna3_wmma_shim.h" "$PT/multihead_fast_rdna3.hip" &
   wait
   build_one deep_fast            -- "$PT/rdna3_wmma_shim.h" "$PT/deep_fast_rdna3.hip"
   ;;
esac
