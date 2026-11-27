import os, sys, re, json, html, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(HERE, "tflite2c.py")).read().replace("\nmain()\n", "\n")
_tmp = os.path.join(HERE, "_tfl_lib.py")
open(_tmp, "w").write(_src)
_spec = importlib.util.spec_from_file_location("tfl", _tmp)
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

SDRAM_BASE = 0xD0000000
ALIGN = 8


def load_plan(path):
    s = open(path, encoding="utf-8").read()
    m = re.search(r"Machine-readable plan.*?<pre>(.*?)</pre>", s, re.S)
    if m:
        return json.loads(html.unescape(m.group(1)))
    return json.loads(s)


def place(G, plan, layers, in_t, out_t):
    nL = len(layers)
    prod = {L["out"]: L["k"] for L in layers}
    last = {}
    for L in layers:
        ts = [L["in0"], L["in1"]] if L["op"] != "NOP" else L["ins"][:1]
        for t in ts:
            if t >= 0:
                last[t] = max(last.get(t, -1), L["k"])
    last[out_t] = nL
    loc, sdoff = {}, {}
    if plan is not None:
        pl = {l["index"]: l for l in plan["layers"]}
        if len(pl) != nL:
            K.die(f"plan has {len(pl)} layers, model has {nL}")
        for L in layers:
            p = pl[L["k"]]
            if p["output"] != L["out"]:
                K.die(f"layer {L['k']}: plan output tensor {p['output']} != model {L['out']}")
            loc[L["out"]] = "SDRAM" if "SDRAM" in str(p["out"]) else "RAM"
        loc[in_t] = "SDRAM" if "SDRAM" in str(plan["input"]["location"]) else "RAM"
        parts = {P["name"]: P for P in plan["partitions"]}
        for sp in plan.get("spilled", []):
            P = parts[sp["partition"]]
            sdoff[int(sp["tensor"])] = int(P["offset"]) + int(sp["offset"])
        if loc[in_t] == "SDRAM":
            P = parts[plan["input"]["location"]]
            sdoff[in_t] = int(P["offset"]) + int(plan["input"]["offset"])
        budget = int(plan["ram_budget"])
    else:
        budget = None
    tensors = sorted(set([in_t] + [L["out"] for L in layers] + [t for L in layers for t in ([L["in0"], L["in1"]] if L["op"] != "NOP" else []) if t >= 0]))
    info = {}
    for t in tensors:
        n = int(np.prod(G.d4(t))) * K.ESZ[G.dtype(t)]
        s0 = prod.get(t, -1)
        e0 = last.get(t, s0)
        info[t] = dict(size=n, s=s0, e=e0, pool=1 if loc.get(t) == "SDRAM" else 0)
        if info[t]["pool"] == 1:
            if t not in sdoff:
                K.die(f"tensor {t} is in SDRAM but the plan gives no offset")
            info[t]["off"] = sdoff[t]
    placed = []
    for t in sorted([t for t in tensors if info[t]["pool"] == 0], key=lambda t: -info[t]["size"]):
        I = info[t]
        sz = (I["size"] + ALIGN - 1) // ALIGN * ALIGN
        conf = sorted((a, b) for (a, b, s, e) in placed if not (I["e"] < s or e < I["s"]))
        off = 0
        for a, b in conf:
            if off + sz <= a:
                break
            off = max(off, b)
        I["off"] = off
        placed.append((off, off + sz, I["s"], I["e"]))
    arena = max([p[1] for p in placed] + [8])
    sd = [t for t in tensors if info[t]["pool"] == 1]
    for i, a in enumerate(sd):
        for b in sd[i + 1:]:
            A, B = info[a], info[b]
            if A["e"] < B["s"] or B["e"] < A["s"]:
                continue
            if A["off"] < B["off"] + B["size"] and B["off"] < A["off"] + A["size"]:
                K.die(f"plan SDRAM tensors {a} and {b} overlap while both alive")
    sd_used = max([info[t]["off"] + info[t]["size"] for t in sd] + [0])
    return tensors, info, arena, sd_used, budget


def emit(name, outdir, G, plan, layers, in_t, out_t, ref, x_in, tensors, info, arena, sd_used, budget, lo, span):
    os.makedirs(outdir, exist_ok=True)
    tid = {t: i for i, t in enumerate(tensors)}
    wbin = bytearray()
    for L in layers:
        if L["w"] is not None:
            wbin += b"\0" * ((-len(wbin)) % 4)
            L["w_off"] = len(wbin)
            L["w_bytes"] = L["w"].size
            wbin += L["w"].tobytes()
        else:
            L["w_off"], L["w_bytes"] = 0, 0
    wbin += b"\0" * ((-len(wbin)) % 4)
    bias_all, mult_all, shift_all, lut_all, lut16_all = [], [], [], [], []
    for L in layers:
        L["qp_idx"] = -1
        L["lut_idx"] = -1
        if L["mult"] is not None:
            L["qp_idx"] = len(mult_all)
            bias_all += L["bias"].tolist()
            mult_all += L["mult"].tolist()
            shift_all += L["shift"].tolist()
        if L["lut"] is not None:
            L["lut_idx"] = len(lut_all)
            lut_all += L["lut"].tolist()
        if L["lut16"] is not None:
            L["lut_idx"] = len(lut16_all)
            lut16_all += L["lut16"].tolist()
        if L["op"] == "MEAN":
            mp = L["mean"]
            L["meanq"] = K.mean_multiplier(mp["sx"], mp["sy"], mp["n"])
            L["p"][2] = mp["n"]
            L["i"] = [mp["zx"], mp["zy"]]
    ops = K.OPS
    h = ["#ifndef NN_MODEL_H", "#define NN_MODEL_H", "#include <stdint.h>", ""]
    h.append(f'#define NN_MODEL_NAME "{name}"')
    h.append(f"#define NN_N_LAYERS {len(layers)}")
    h.append(f"#define NN_N_BUFS {len(tensors)}")
    h.append(f"#define NN_ARENA_BYTES {arena}u")
    h.append(f"#define NN_SRAM_BUDGET {budget if budget is not None else 0}u")
    h.append(f"#define NN_SDRAM_USED {sd_used}u")
    h.append(f"#define NN_IN_BUF {tid[in_t]}")
    h.append(f"#define NN_OUT_BUF {tid[out_t]}")
    h.append(f"#define NN_INPUT_DTYPE {G.dtype(in_t)}")
    h.append(f"#define NN_INPUT_SEED {K.LCG_SEED}u")
    h.append(f"#define NN_INPUT_LO {float(np.float32(lo)).hex()}f")
    h.append(f"#define NN_INPUT_SPAN {float(np.float32(span)).hex()}f")
    h.append(f"#define NN_WEIGHTS_BYTES {len(wbin)}u")
    h.append("")
    h.append("enum { " + ", ".join(f"NN_OP_{n} = {i}" for i, n in enumerate(ops)) + " };")
    h.append("enum { NN_DT_I8 = 0, NN_DT_F32 = 1, NN_DT_I32 = 2, NN_DT_U8 = 3 };")
    h.append("enum { NN_POOL_SRAM = 0, NN_POOL_SDRAM = 1 };")
    h.append("")
    h.append("typedef struct {\n    uint32_t off;\n    uint32_t bytes;\n    uint16_t h, w, c;\n    uint8_t dtype, pool;\n} nn_buf_t;\n")
    h.append("typedef struct {\n    uint8_t op;\n    int16_t in0, in1, out;\n    int16_t kh, kw, sh, sw, dh, dw, pt, pl, pb, pr;\n"
             "    int16_t act_min, act_max;\n    int16_t p[3];\n    int16_t bh, bw, bc;\n    int32_t qp, lut;\n"
             "    uint32_t w_off, w_bytes;\n    int32_t i0, i1, m, s;\n    float f0, f1;\n} nn_layer_t;\n")
    h.append("extern const nn_buf_t nn_bufs[NN_N_BUFS];")
    h.append("extern const nn_layer_t nn_layers[NN_N_LAYERS];")
    h.append("extern const char *const nn_layer_names[NN_N_LAYERS];")
    h.append("extern const int32_t nn_bias[];\nextern const int32_t nn_mult[];\nextern const int8_t nn_shift[];")
    h.append("extern const int8_t nn_lut[];\nextern const uint16_t nn_lut16[];")
    h.append("extern const int8_t nn_weights[];")
    h.append("extern const int64_t nn_ref_sum[NN_N_LAYERS];")
    h.append(f"#define NN_REF_OUT_BYTES {ref[out_t].nbytes}u")
    h.append("extern const uint8_t nn_ref_out[NN_REF_OUT_BYTES];")
    h.append("\n#endif\n")
    open(os.path.join(outdir, "nn_model.h"), "w").write("\n".join(h))

    c = ['#include "nn_model.h"\n', "const nn_buf_t nn_bufs[NN_N_BUFS] = {"]
    for t in tensors:
        d = G.d4(t)
        I = info[t]
        c.append(f"    {{ {I['off']}u, {I['size']}u, {d[1]}, {d[2]}, {d[3]}, {G.dtype(t)}, {I['pool']} }},")
    c.append("};\n\nconst nn_layer_t nn_layers[NN_N_LAYERS] = {")
    for L in layers:
        in0 = tid[L["in0"]] if L["in0"] >= 0 and L["op"] != "NOP" else -1
        in1 = tid[L["in1"]] if L["in1"] >= 0 else -1
        out = tid[L["out"]]
        m_, s_ = L.get("meanq", (0, 0))
        b = L["bshape"]
        c.append(f"    {{ NN_OP_{L['op']}, {in0}, {in1}, {out}, {L['kh']}, {L['kw']}, {L['sh']}, {L['sw']}, "
                 f"{L['dh']}, {L['dw']}, {L['pt']}, {L['pl']}, {L['pb']}, {L['pr']}, {L['act_min']}, {L['act_max']}, "
                 f"{{ {L['p'][0]}, {L['p'][1]}, {L['p'][2]} }}, {b[0]}, {b[1]}, {b[2]}, {L['qp_idx']}, {L['lut_idx']}, "
                 f"{L['w_off']}u, {L['w_bytes']}u, {L['i'][0]}, {L['i'][1]}, {m_}, {s_}, {float(L['f'][0])!r}f, {float(L['f'][1])!r}f }},")
    c.append("};\n\nconst char *const nn_layer_names[NN_N_LAYERS] = {")
    for L in layers:
        c.append(f'    "{L["name"]}",')
    c.append("};\n")

    def arr(nm, typ, vals, per=12):
        vals = vals if vals else [0]
        c.append(f"const {typ} {nm}[] = {{")
        for i in range(0, len(vals), per):
            c.append("    " + ", ".join(str(int(v)) for v in vals[i:i + per]) + ",")
        c.append("};\n")

    arr("nn_bias", "int32_t", bias_all, 8)
    arr("nn_mult", "int32_t", mult_all, 8)
    arr("nn_shift", "int8_t", shift_all, 16)
    arr("nn_lut", "int8_t", lut_all, 16)
    arr("nn_lut16", "uint16_t", lut16_all, 16)
    sums = []
    for L in layers:
        v = ref[L["out"]]
        sums.append(int(np.sum(v.astype(np.int64))) if v.dtype in (np.int8, np.uint8, np.int32) else 0)
    c.append("const int64_t nn_ref_sum[NN_N_LAYERS] = {")
    for i in range(0, len(sums), 6):
        c.append("    " + ", ".join(f"{v}LL" for v in sums[i:i + 6]) + ",")
    c.append("};\n")
    ob = np.frombuffer(ref[out_t].tobytes(), dtype=np.uint8)
    arr("nn_ref_out", "uint8_t", ob.tolist(), 16)
    open(os.path.join(outdir, "nn_model.c"), "w").write("\n".join(c))

    a8 = np.frombuffer(bytes(wbin), dtype=np.int8)
    with open(os.path.join(outdir, "nn_weights.c"), "w") as f:
        f.write('#include "nn_model.h"\n\n__attribute__((aligned(4))) const int8_t nn_weights[NN_WEIGHTS_BYTES] = {\n')
        for i in range(0, len(a8), 32):
            f.write("    " + ",".join(map(str, a8[i:i + 32].tolist())) + ",\n")
        f.write("};\n")
    return len(wbin)


def generate(tfl, plan_path, outdir):
    K.PLAN_PATH = ""
    G, _, _, layers, in_t, out_t = K.build(tfl)
    plan = load_plan(plan_path) if plan_path else None
    x_in, lo, hi = K.make_input(G, in_t, layers)
    ref, out_final = K.reference_run(G, layers, x_in)
    ok, rep = K.run_numpy(G, layers, x_in, ref, isolated=True, in_t=in_t)
    ok2, _ = K.run_numpy(G, layers, x_in, ref, isolated=False, in_t=in_t)
    tensors, info, arena, sd_used, budget = place(G, plan, layers, in_t, out_t)
    name = os.path.splitext(os.path.basename(tfl))[0]
    wb = emit(name, outdir, G, plan, layers, in_t, out_t, ref, x_in, tensors, info, arena, sd_used, budget, lo, hi - lo)
    n_sd = sum(1 for t in tensors if info[t]["pool"] == 1)
    print(f"  plan {os.path.basename(plan_path) if plan_path else '-'}")
    print(f"  numpy vs TFLite: each layer {'MATCH' if ok else 'MISMATCH'}, full chain {'MATCH' if ok2 else 'MISMATCH'}")
    if not ok:
        print("\n".join(rep))
    print(f"  SRAM arena {arena} B (budget {budget}), SDRAM used {sd_used} B, tensors in SDRAM {n_sd}/{len(tensors)}, weights {wb} B")
    if budget is not None and arena > budget:
        print("  WARNING: arena larger than the plan's SRAM budget")
    return ok and ok2


if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None, sys.argv[3])
