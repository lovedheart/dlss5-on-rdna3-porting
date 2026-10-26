#!/usr/bin/env python3
# e2e synthetic weight assets. hip_reference_network.h ReadWeights falls back to
# <name minus .f32>.f16 (halves) when the .f32 file is absent, so every weight is
# emitted as compact 2-byte halves with the EXACT element count from WeightElements().
# Values: benign halves alternating +0.375 (0x33C0) / -0.375 (0xB3C0); scale-ish
# files use 1.0 (0x3C00). Numbers are meaningless by design; timing is not.
import os, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)

def elems(name):
    if name == "head-matrix.f32": return 524288
    if name == "decoder39-weights.f32": return 524800
    if name == "post70-scales.f32": return 64
    if name == "post70-head.f32": return 96
    b = int(name[5:name.index('-')]); typ = name[name.index('-')+1:]
    c = 32 if b <= 4 else 64 if b <= 8 else 128 if b <= 14 else 256 if b <= 22 else 512 if b <= 30 else 1024 if b <= 38 else 512 if b <= 47 else 256 if b <= 55 else 128 if b <= 61 else 64 if b <= 65 else 32
    t = {"ds.f32": 2*c*c, "weights.f32": 2*c*c+c, "ffwd.f32": 524288, "ffwd-projection.f32": 262656,
         "ffn.f32": 8736 if c == 32 else 9*c*c+c, "attention.f32": 8225 if c == 32 else 4*c*c+(c//32)*4096+c//32+c,
         "expand.f32": 4194304, "contract.f32": 4195328, "qkv.f32": 3145760, "projection.f32": 1049600}[typ]
    return t

def write(name):
    n = elems(name)
    scale = name.endswith("scales.f32") or name == "post70-head.f32" or name.endswith("ffwd-projection.f32")
    pat = (b"\x00\x3c" if scale else b"\xc0\x33\xc0\xb3")
    body = (pat * (n // len(pat) + 1))[: (n * 2) - (n % 2)] if False else None
    if scale or name.endswith("weights.f32"):
        data = (b"\x00\x3c" * n)[: 2 * n]
    else:
        data = (b"\xc0\x33" * (n + 1))[: 2 * n]
        # alternate sign every element: rewrite odd halves to 0xB3C0
        ba = bytearray(data)
        ba[2::4] = b"\x33" * (len(ba[2::4])); ba[3::4] = b"\xb3" * (len(ba[3::4]))
        ba[0::4] = b"\xc0" * (len(ba[0::4])); ba[1::4] = b"\x33" * (len(ba[1::4]))
        data = bytes(ba)
    with open(os.path.join(OUT, name[:-4] + ".f16"), "wb") as f:
        f.write(data)
    return n

names = []
for b in range(70):
    if b <= 4 or b >= 66: names += [f"block{b}-ffn.f32", f"block{b}-attention.f32"]
    elif 5 <= b <= 30 or 40 <= b <= 65:
        names += [f"block{b}-ffn.f32", f"block{b}-attention.f32"]
        if b in (23,24,25,26,27,28,29,30) or 40 <= b <= 47:
            names += [f"block{b}-ffwd.f32", f"block{b}-ffwd-projection.f32"]
    elif 31 <= b <= 38: names += [f"block{b}-expand.f32", f"block{b}-contract.f32", f"block{b}-qkv.f32", f"block{b}-projection.f32"]
    if b in (4,8,14,22,30,47,55,61,65): names.append(f"block{b}-ds.f32")
    if b in (39,48,56,62,66): names.append(f"block{b}-weights.f32")
names += ["decoder39-weights.f32", "head-matrix.f32", "post70-scales.f32", "post70-head.f32", "post70-attention.f32", "post70-ffn.f32"]
tot = 0
for nm in names: tot += write(nm)
print(f"{len(names)} weight files, {tot} elements, {os.popen('du -sh '+OUT).read().split()[0]} on disk")
