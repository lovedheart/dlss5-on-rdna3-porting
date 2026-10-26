# Linux / gfx1103 port patches applied to the harness (repo copies only — the source repo was never modified)

All files here are copies of `Development/HIP/*` from `/home/lovedheart/dlss5-on-amd-9070xt-porting` (read-only), patched as follows.

## hip_api.h
1. `#include <windows.h>` → on non-Windows: `#include <dlfcn.h>`, `HMODULE=void*`, `GetProcAddress→dlsym` (e2e patch #1).
2. `Api::Api` DLL load: `LoadLibraryExW(amdhip64_7.dll)` → `dlopen` with fallbacks `libamdhip64.so.7` → `libamdhip64.so` → `libamdhip64.so.6` (e2e patch #2). All HIP entry points are still resolved by `dlsym` exactly like the Windows `GetProcAddress` path; the function-pointer ABI types are already plain C types from the rocm-7.1.1 headers, no struct changes needed. `hipGetDevicePropertiesR0600` exists in this ROCm's `libamdhip64.so.7`.
3. Stubs for diagnostics that touch Windows: `OutputDebugStringA` → no-op, `VK_F8` defined, `GetAsyncKeyState` → 0 (the F8 ViT hotkey can never fire) (e2e patch #5).

## hip_reference_network.h
4. `#include "../../src/*.h"` → `#include "src/*.h"` (headers copied into `harness/src/`) (e2e patch #3). No logic changes.
   Device selection: this file has no LUID logic; the only `gcnArchName` use (`CompactVitTailDeviceEligible`, gfx1201+9070-XT check) correctly evaluates false on gfx1103. `opt.device=0` is used as-is.

## device_network.cpp
5. e2e patch #4 — the original sets `o.wmma=o.wave=o.pooled=true`. Two facts from the unmodified harness code:
   - `opt.wmma=true` redirects every accelerated GEMM launch to modules `c32_wmma`/`mh_wmma`/`deep_wmma`, which this offline harness never loads (no `LoadModule` call site exists for them) → `map::at` throw.
   - `opt.wave=true` redirects `c32_attn_normalize`/`mh_normalize`/... to module `wave` (wave-pointwise.hsaco), likewise never loaded here.
   So the original flag set cannot run even on Windows with this driver. Replacement: `pooled=true` (memory pooling only) and a config switch `E2E_CONFIG`:
   - `ref`   — 71-block schedule on the 4 reference modules (unchanged repo kernels, compiled for gfx1103)
   - `tiled` — same schedule with the tiled GEMM ports (`opt.tiled`, module `c32_tiled`+`multihead-tiled`)
   - `fast`  — same schedule on the production fast-family ports (`fast_c32/fast_mh/fast_deep/fast_prefix`, `fast_vit` on the 900 run)
   `fast_vit` (900 mode) is only enabled in the `fast` config because the blocked ViT kernels live in `deep_fast`.
6. Environment used at run time: `DLSS5_MULTI_PASS=1` (Original mode, single network evaluation per frame, matching the "71-block" schedule; without it the default would attempt the Small model and log a fallback), `DLSS5_MULTI_PASS_PREDICT=0` (skips the optional `multi-pass-predict.hsaco` gain-predictor module, not part of the 71-block network).

## Runtime trace (no code change)
Built with `-DHIP_SWIN_PERSISTENT_TRACE`, which activates the harness's own `TOPO,module,kernel,count,groups,threads` printf already present in `Run()`. This gives the per-frame launch schedule without adding instructions to the timed path beyond a `printf` per launch.

## build-modules.sh
Replicates `build-modules.ps1`'s gfx1103 rows (defines prepended, shim+port concatenated) and the 3-stage compile from `Development/tools/llvm-fork/compile-modules.py`:
`clang --offload-device-only -emit-llvm` → `clang -target amdgcn-amd-amdhsa -mcpu=gfx1103 -mllvm -amdgpu-internalize-symbols` → `ld.lld --no-undefined -shared` (the internalize+lld link is required; unlinked carve-outs load but run 8–50× slow through the module API).
Added one extra module not in the gfx1103 map: `boundary-fast` (hard-required by `opt.fast_c32`), composed as shim + `c32_fast_attention_rdna3.hip` + repo `boundary_fast.hip` with the single `__builtin_amdgcn_cvt_f32_fp8(` → `rdna3::cvt_f32_fp8(` sed (same transformation family as the shipped ports).

## Synthetic assets (gen_assets.py)
Real weights are not on this machine. Every weight is a synthetic `.f16` (ReadWeights fallback) with the exact `WeightElements(name)` count, filled with ±0.375 half patterns (scales/head files: 1.0). Numerical output is meaningless; kernel geometry, instruction mix, memory traffic and launch counts are exactly what the schedule produces. See REPORT.md caveats.

## Patch #8 / #8b — optional per-launch timing instrumentation (hip_reference_network.h)
`E2E_TIME=1` wraps each `hipLaunchKernel` in `Run()` with `hipEventCreate/Record/ElapsedTime/Synchronize` and prints `TIMED,us,module,kernel`. First version kept the previous-launch timestamp in a `static` local inside the variadic-templated `Run()` — `static` there is per template instantiation, so deltas spanned whole frames. Fixed in #8b by moving the state to a namespace-scope accessor `E2eLastTime()`. Sum of per-launch times now matches `wall_ms` to <1%, validating the attribution. `E2E_SYNC=1` adds a full `StreamSync` per launch for crash localization. Neither is enabled in the headline (uninstrumented) timing runs; instrumented runs are reported alongside and agree within noise.

## Unit note
`device_network` prints `wall_ms=` values that are actually **microseconds** (chrono count formatted with the ms label). All tables in REPORT.md are corrected to real ms.
