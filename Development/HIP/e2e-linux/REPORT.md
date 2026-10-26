# DLSS5-AMD 71-block neural renderer — end-to-end per-frame GPU time on gfx1103 (Radeon 780M)

Measured with the offline HIP harness (`device_network`) copied out of the strictly read-only
source repo, running ported gfx1103 kernel modules. All artifacts live in this `e2e/` directory.

**Bottom line (Original / 71-block schedule, 3 iterations, warm frames):**

| Config | Geometry | best ms/frame | median ms/frame | fps (best / median) |
|---|---|---|---|---|
| **fast** (production fast-family ports) | 512×512 | **1.93** | 1.94 | **518 / 516** |
| **fast** | 1600×1024 ("900") | **11.73** | 12.12 | **85.2 / 82.5** |
| tiled (tiled GEMM ports) | 512×512 | 6.38 | 6.52 | 157 / 153 |
| tiled | 1600×1024 | 35.7 | 36.7 | 28 / 27 |
| ref (unmodified repo kernels, fp32) | 512×512 | 8.90 | 8.93 | 112 / 112 |
| ref | 1600×1024 | 43.7 | 45.5 | 23 / 22 |

Note: the harness's `wall_ms=` field is actually **microseconds** (formatting quirk in the
original print); the table above is corrected to real milliseconds. Runs on this machine
share the GPU with the desktop — iteration-to-iteration variance (especially on the cold
first iteration) can be several-fold; best/median of the warm iterations is quoted.

Raw logs: `run_{ref,tiled,fast}{512,900}.log`, instrumented reruns
`run_fast{512,900}_timed.log`.

## 1. Modules built (all gfx1103, ELF code objects loadable via hipModuleLoadData)

Built by `build-modules.sh` (3-stage recipe: device-only LLVM bitcode → `clang
-target amdgcn-amd-amdhsa -mcpu=gfx1103 -mllvm -amdgpu-internalize-symbols` → `ld.lld
--no-undefined -shared`; replicating `build-modules.ps1` gfx1103 rows + `compile-modules.py`).
Verified loadable + launchable with `loadprobe`.

| stem.hsaco | size | source |
|---|---|---|
| c32_prefix_reference | 30 KB | repo `hip/c32_prefix_reference.hip` (unmodified) |
| multihead-reference | 59 KB | repo `hip/multihead_reference.hip` (unmodified) |
| deep_reference | 61 KB | repo `hip/deep_reference.hip` (unmodified) |
| boundary_reference | 15 KB | repo `hip/boundary_reference.hip` (unmodified) |
| c32_tiled | 415 KB | proto `c32_tiled_1103.hip` (512-threads variant) + shim |
| multihead-tiled | 442 KB | proto `multihead_tiled_1103.hip` (512-threads variant) + shim |
| c32_fast | 256 KB | patchtree `c32_fast_rdna3.hip` + shim |
| c32_fast_attention | 82 KB | patchtree `c32_fast_attention_rdna3.hip` + shim (separate TU) |
| boundary-fast | 103 KB | shim + attention TU + repo `boundary_fast.hip` (fp8 builtin → `rdna3::cvt_f32_fp8` sed) |
| multihead-fast | 262 KB | patchtree `multihead_fast_rdna3.hip` + shim |
| multihead-fast-packed | 247 KB | same TU with packed-weight defines |
| deep_fast | 3.0 MB | patchtree `deep_fast_rdna3.hip` + shim (~4 min compile) |
| prefix_fast | 82 KB | patchtree prefix port + shim |

## 2. What actually launches per frame (fast config, 780 launches/frame)

Full census: `schedule_fast512.txt` (from the harness's own `HIP_SWIN_PERSISTENT_TRACE` launch
logging; identical 780/frame at 900p). Families: mh_fast 425, deep_fast 133, mh (glue:
shift_pack/crop/pool) 117, c32_fast_attention 60, c32_fast_ffn 20, boundary_fast 11,
boundary 10, prefix_fast 2, deep (vit_gather) 2.

Missing exports: **zero.** A `hipModuleGetFunction` census over every (module, kernel) the fast
schedule references (HASMISS pass, `trace_fast512.err`) reported 0 misses — every kernel the
71-block Original schedule needs is exported by the built modules.

Fallbacks inside the frame: none. See PATCHES.md for the two config-level deviations required
to make the schedule runnable at all (the shipped `o.wmma=o.wave=true` flags target modules
this offline harness never loads, and the optional `multi-pass-predict` module is disabled via
env — it is outside the 71-block network).

## 3. Per-stage split (fast config, per-launch hipEvent timing; `breakdown_fast{512,900}.txt`)

512×512 frame = 1.72 ms instrumented (matches the uninstrumented 1.93 ms within noise):

| module | ms/frame @512 | ms/frame @900 |
|---|---|---|
| c32_fast_ffn (full-res FFN GEMMs) | 0.99 (57%) | 6.30 (58%) |
| mh_fast (multihead attention/FFN) | 0.43 (25%) | 2.70 (25%) |
| c32_fast_attention (conv3x2 attn) | 0.18 (10%) | 1.17 (11%) |
| deep_fast (ViT/split/decoder) | 0.11 (6%) | 0.63 (6%) |
| glue (mh, boundary*, prefix_fast, deep) | 0.02 (1%) | 0.12 (1%) |

Dominant kernels @512: `c32_ffn_contract_fast_map3` 0.52 ms (6 launches, full-res 128-ch FFN),
`c32_ffn_contract_fast` 0.41 ms (4), `mh_qkv_normalize_fast` 0.20 ms (52×~4 µs). The 512→900
scaling is almost perfectly linear in pixel count (×6.25 pixels → ×6.1 time), consistent with
bandwidth-bound full-resolution GEMMs.

## 4. How to reproduce

```
DLSS5_MULTI_PASS=1 DLSS5_MULTI_PASS_PREDICT=0 E2E_CONFIG=fast \
  ./device_network assets modules data/in512.bin data/noise.bin data/out512.bin
# append "900" for 1600x1024; E2E_CONFIG=ref|tiled for the other configs
# optional: E2E_TRACE=1 (launch trace), E2E_TIME=1 (per-launch µs), E2E_SYNC=1 (crash debug)
```
Build: `g++ -O2 -std=c++17 -DHIP_SWIN_PERSISTENT_TRACE -o device_network harness/device_network.cpp -ldl`
Modules: `./build-modules.sh` (gfx1103 only). Assets: `python3 gen_assets.py`.

## 5. Deviations and what this does / does not prove

1. **Synthetic weights** (206 `.f16` files, ±0.375 halves; `gen_assets.py`). Real weights don't
   exist on this machine. Proves: exact launch schedule, correct kernel execution on gfx1103
   ISA, memory-traffic-dominated timing (these GEMMs are data-movement bound; weight *values*
   barely change the instruction mix — the ports are f32/fp16 WMMA paths without
   data-dependent control flow). Does NOT prove: output quality, or exclusion of
   value-dependent costs (none expected). Output was checked: all finite (saturated to 1.0,
   as expected for consistent ±0.375 products), no NaN/inf.
2. **Linux shims** (`hip_api.h`: dlopen of libamdhip64 replaces LoadLibrary; submit_pulse path
   inert). Identical HIP driver-API calls underneath; host-side only, no timing impact.
3. **E2E_CONFIG replaces the shipped `wmma=wave=true` flags**, which cannot run with this
   offline driver on any platform (the harness never loads `*_wmma`/`wave` modules → map::at
   throw). The three reported configs are the runnable flag sets: reference-only, tiled ports,
   and the production fast-family ports. The `fast` config is the meaningful shipping number.
4. **`fast_vit` enabled in fast config** (both geometries): `deep_fast` exports only the
   `*_fast` ViT kernels; the non-fast-vit ViT path needs plain-named kernels that live in
   `deep_reference`, and mixing would mispair producer/consumer layouts. With `fast_deep`,
   `fast_vit` is the intended pairing.
5. **DLSS5_MULTI_PASS=1** selects the Original single-network schedule (the 71-block frame);
   PREDICT=0 skips an optional gain predictor outside the 71 blocks.
6. **Tiled modules rebuilt from proto sources**: patchtree tiled ports declare tile attribute
   (256,256) but the harness launches 512 threads → launch failure; proto sources use
   (512,512) and match. (Tiled config is secondary; fast is the headline.)
7. **GPU is a shared iGPU** (desktop compositor active). Numbers are conservative; the
   tightest warm-iteration best is quoted alongside the median.
