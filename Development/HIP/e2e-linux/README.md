# gfx1103 end-to-end harness (Linux)

Runs the repo's offline 71-block reference network (device_network +
hip_reference_network.h) on Linux + gfx1103 against the ported .hsaco
modules from `build-modules.sh` (same recipes as build-modules.ps1 gfx1103
rows). Synthetic .f16 weights via gen_assets.py (exact WeightElements()
sizes; timing-valid, numerics meaningless). Measured on Radeon 780M:
512x512 = 1.9 ms/frame, 900p = 11.7 ms/frame. See REPORT.md; every source
deviation from Development/HIP/* documented in PATCHES.md.

Build: hipcc -O2 device_network.cpp -o device_network
Run:   DLSS5_MULTI_PASS=1 DLSS5_MULTI_PASS_PREDICT=0 E2E_CONFIG=fast \
       ./device_network assets modules in_rgba.bin noise.bin out.bin [900]
