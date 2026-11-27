import os, sys, re
import numpy as np
import placement as H

K = H.K


def main(tfl, plan_path, outdir, zip_model_c):
    K.PLAN_PATH = ""
    G, _, _, layers, in_t, out_t = K.build(tfl)
    plan = H.load_plan(plan_path)
    tensors, info, arena, sd_used, budget = H.place(G, plan, layers, in_t, out_t)
    src = open(zip_model_c).read()
    body = src.split("nn_bufs[NN_N_BUFS] = {")[1].split("};")[0]
    rows = re.findall(r"\{([^}]*)\}", body)
    if len(rows) != len(tensors):
        K.die(f"zip has {len(rows)} buffers, model has {len(tensors)}")
    for t, r in zip(tensors, rows):
        v = [x.strip() for x in r.split(",")]
        d = G.d4(t)
        if (int(v[2]), int(v[3]), int(v[4])) != (d[1], d[2], d[3]):
            K.die(f"buffer for tensor {t} shape mismatch")
    S = int(budget)
    os.makedirs(outdir, exist_ok=True)
    n_sd = sum(info[t]["pool"] for t in tensors)
    sram_b = sum(info[t]["size"] for t in tensors if info[t]["pool"] == 0)
    sd_b = sum(info[t]["size"] for t in tensors if info[t]["pool"] == 1)
    h = ["#ifndef NN_PLACE_H", "#define NN_PLACE_H", "#include <stdint.h>", "#include \"nn_model.h\"",
         f"#define NN_PLAN_SRAM_KB {S // 1024}u",
         f"#define NN_SRAM_ARENA_BYTES {max(arena, 32) if n_sd < len(tensors) else 32}u",
         f"#define NN_SDRAM_EXTENT {sd_used}u",
         f"#define NN_ACT_SRAM_BYTES {sram_b}u",
         f"#define NN_ACT_SDRAM_BYTES {sd_b}u",
         f"#define NN_N_SDRAM_BUFS {n_sd}u",
         "enum { NN_POOL_SRAM = 0, NN_POOL_SDRAM = 1 };",
         "typedef struct { uint8_t pool; uint32_t off; uint32_t bytes; } nn_place_t;",
         "extern const nn_place_t nn_place[NN_N_BUFS];", "extern const int16_t nn_in_zp[NN_N_LAYERS];", "extern const int16_t nn_out_zp[NN_N_LAYERS];", "#endif", ""]
    open(os.path.join(outdir, "nn_place.h"), "w").write("\n".join(h))
    c = ['#include "nn_place.h"', "", "const nn_place_t nn_place[NN_N_BUFS] = {"]
    for t in tensors:
        I = info[t]
        c.append(f"    {{ {I['pool']}, {I['off']}u, {I['size']}u }},")
    c += ["};", "", "const int16_t nn_in_zp[NN_N_LAYERS] = {"]
    zs = [int(G.q(L["in0"])[1][0]) if L["in0"] >= 0 else 0 for L in layers]
    for i in range(0, len(zs), 16):
        c.append("    " + ", ".join(map(str, zs[i:i + 16])) + ",")
    c += ["};", "", "const int16_t nn_out_zp[NN_N_LAYERS] = {"]
    zo = [int(G.q(L["out"])[1][0]) for L in layers]
    for i in range(0, len(zo), 16):
        c.append("    " + ", ".join(map(str, zo[i:i + 16])) + ",")
    c += ["};", ""]
    open(os.path.join(outdir, "nn_place.c"), "w").write("\n".join(c))
    print(f"S{S // 1024}: SRAM arena {arena} B (budget {S}), SDRAM extent {sd_used} B, "
          f"buffers in SDRAM {n_sd}/{len(tensors)}, act bytes SRAM {sram_b} SDRAM {sd_b}")
    if arena > S and n_sd < len(tensors):
        print("  WARNING: arena larger than budget")


if __name__ == "__main__":
    main(*sys.argv[1:5])
