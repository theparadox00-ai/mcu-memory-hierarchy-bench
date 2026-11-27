import os, sys, json, math, glob
import numpy as np

TFLITE_PATHS = {}
PLAN_PATH = None
OUT_DIR = None
WEIGHTS_IN_FLASH = True
FLASH_BYTES = 512 * 1024
SECTOR = 512
LCG_SEED = 12345

try:
    from tensorflow.lite.python import schema_py_generated as fb
except Exception:
    from ai_edge_litert import schema_py_generated as fb
import tensorflow as tf

OPNAME = {v: k for k, v in fb.BuiltinOperator.__dict__.items() if not k.startswith("_")}
DT_I8, DT_F32, DT_I32, DT_U8 = 0, 1, 2, 3
TT = {9: DT_I8, 0: DT_F32, 2: DT_I32, 3: DT_U8}
ESZ = {DT_I8: 1, DT_F32: 4, DT_I32: 4, DT_U8: 1}

OPS = ["NOP", "QUANTIZE", "DEQUANTIZE", "CONV", "DWCONV", "FC", "MAXPOOL", "AVGPOOL", "PAD",
       "RESHAPE", "MEAN", "LOGISTIC", "ADD", "SOFTMAX", "MUL"]
OPID = {n: i for i, n in enumerate(OPS)}
MAP = {"QUANTIZE": "QUANTIZE", "DEQUANTIZE": "DEQUANTIZE", "CONV_2D": "CONV",
       "DEPTHWISE_CONV_2D": "DWCONV", "FULLY_CONNECTED": "FC", "MAX_POOL_2D": "MAXPOOL",
"PAD": "PAD", "PADV2": "PAD", "RESHAPE": "RESHAPE",
       "MEAN": "MEAN", "LOGISTIC": "LOGISTIC", "AVERAGE_POOL_2D": "AVGPOOL", "ADD": "ADD", "MUL": "MUL", "SOFTMAX": "SOFTMAX", "SHAPE": "NOP", "STRIDED_SLICE": "NOP",
       "PACK": "NOP"}




def die(msg):
    raise SystemExit("ERROR: " + msg)


def quantize_multiplier(d):
    if d == 0.0:
        return 0, 0
    q, shift = math.frexp(d)
    qf = int(round(q * (1 << 31)))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    if shift > 30:
        shift, qf = 30, (1 << 31) - 1
    return qf, shift


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


ROUND_UP = {"CONV": True, "DWCONV": True, "FC": True, "ADD": False, "MEAN": False, "MUL": False, "SOFTMAX": False}


def mbqm(acc, mult, shift, up=False):
    acc = np.asarray(acc, dtype=np.int64)
    mult = np.asarray(mult, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    x = acc * (np.int64(1) << left)
    ab = x * mult
    nudge = np.where(ab >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    s = ab + nudge
    hi = np.where(s >= 0, s >> 31, -((-s) >> 31))
    hi = np.clip(hi, -(1 << 31), (1 << 31) - 1)
    mask = (np.int64(1) << right) - 1
    rem = hi & mask
    thr = (mask >> 1) + (0 if up else (hi < 0))
    return (hi >> right) + (rem > thr)


def mean_multiplier(sx, sy, n):
    m0, s0 = quantize_multiplier(float(sx) / float(sy))
    sh = n.bit_length() - 1
    sh = min(sh, 32, 31 + s0)
    return int((m0 << sh) // n), s0 - sh


def act_range(act, scale, zp):
    lo, hi = -128, 127
    if act == 1:
        lo = max(lo, zp)
    elif act == 3:
        lo = max(lo, zp)
        hi = min(hi, zp + int(round(6.0 / scale)))
    elif act == 2:
        lo = max(lo, zp + int(round(-1.0 / scale)))
        hi = min(hi, zp + int(round(1.0 / scale)))
    return lo, hi


def same_pad(inp, out, k, s, d):
    eff = (k - 1) * d + 1
    tot = max((out - 1) * s + eff - inp, 0)
    return tot // 2, tot - tot // 2


def load_plan(tfl):
    p = PLAN_PATH
    if p is None:
        stem = os.path.splitext(tfl)[0]
        hits = sorted(glob.glob(stem + "_plan*.json"))
        p = hits[0] if hits else None
    if not p:
        print("  no plan JSON: every layer runs whole")
        return None, None
    print(f"  plan: {p}")
    return json.load(open(p)), p


class Gen:
    def __init__(self, path):
        self.path = path
        self.buf = open(path, "rb").read()
        self.m = fb.ModelT.InitFromObj(fb.Model.GetRootAsModel(self.buf, 0))
        if len(self.m.subgraphs) != 1:
            die("only one subgraph supported")
        self.g = self.m.subgraphs[0]

    def t(self, i):
        return self.g.tensors[i]

    def shape(self, i):
        s = self.t(i).shape
        return [int(v) for v in s] if s is not None else []

    def d4(self, i):
        s = self.shape(i)
        if len(s) == 0:
            return [1, 1, 1, 1]
        if len(s) == 2:
            return [1, 1, 1, s[1]] if s[0] == 1 else [1, 1, s[0], s[1]]
        if len(s) == 1:
            return [1, 1, 1, s[0]]
        if len(s) == 3:
            return [1] + s
        return s

    def data(self, i):
        b = self.m.buffers[self.t(i).buffer]
        if b is None or b.data is None or len(b.data) == 0:
            return None
        return np.asarray(b.data, dtype=np.uint8).tobytes()

    def q(self, i):
        qp = self.t(i).quantization
        if qp is None or qp.scale is None or len(qp.scale) == 0:
            return np.array([1.0], np.float32), np.array([0], np.int64)
        return np.asarray(qp.scale, np.float32), np.asarray(qp.zeroPoint, np.int64)

    def dtype(self, i):
        tt = self.t(i).type
        if tt not in TT:
            die(f"tensor {i}: unsupported type {tt}")
        return TT[tt]


def build(tfl):
    if not os.path.isfile(tfl):
        die(f"{tfl} not found")
    print(f"\n=== {os.path.basename(tfl)} ===")
    G = Gen(tfl)
    plan, plan_path = load_plan(tfl)
    g = G.g

    layers = []
    for k, op in enumerate(g.operators):
        code = G.m.operatorCodes[op.opcodeIndex]
        name = OPNAME[max(code.builtinCode, code.deprecatedBuiltinCode)]
        if name not in MAP:
            die(f"layer {k}: op {name} not supported")
        ins = [int(v) for v in op.inputs]
        outs = [int(v) for v in op.outputs]
        L = dict(k=k, name=name, op=MAP[name], ins=ins, out=outs[0], o=op.builtinOptions,
                 in0=ins[0] if ins else -1, in1=-1, kh=1, kw=1, sh=1, sw=1, dh=1, dw=1,
                 pt=0, pl=0, pb=0, pr=0, act_min=-128, act_max=127, w=None, wshape=None,
                 bias=None, mult=None, shift=None, lut=None, lut16=None, bshape=[1, 1, 1], mean=None, p=[0] * 8, f=[0.0, 0.0],
                 i=[0, 0])
        layers.append(L)

    in_t = int(g.inputs[0])
    out_t = int(g.outputs[0])

    for L in layers:
        o, op = L["o"], L["op"]
        x, y = L["in0"], L["out"]
        if op in ("CONV", "DWCONV"):
            w_t = L["ins"][1]
            b_t = L["ins"][2] if len(L["ins"]) > 2 else -1
            xi, yo = G.d4(x), G.d4(y)
            ws = G.shape(w_t)
            KH, KW = ws[1], ws[2]
            L.update(kh=KH, kw=KW, sh=o.strideH, sw=o.strideW, dh=o.dilationHFactor or 1,
                     dw=o.dilationWFactor or 1)
            if o.padding == 0:
                L["pt"], L["pb"] = same_pad(xi[1], yo[1], KH, o.strideH, L["dh"])
                L["pl"], L["pr"] = same_pad(xi[2], yo[2], KW, o.strideW, L["dw"])
            if op == "DWCONV":
                L["p"][0] = yo[3] // xi[3]
            sx, zx = G.q(x)
            sw_, zw = G.q(w_t)
            sy, zy = G.q(y)
            if np.any(zw != 0):
                die(f"layer {L['k']}: weight zero point must be 0")
            W = np.frombuffer(G.data(w_t), dtype=np.int8).reshape(ws)
            L["w"], L["wshape"] = W, ws
            OC = yo[3]
            swv = np.broadcast_to(sw_, (OC,)) if sw_.size == 1 else sw_
            L["bias"] = (np.frombuffer(G.data(b_t), dtype=np.int32).copy() if b_t >= 0 and G.data(b_t)
                         else np.zeros(OC, np.int32))
            mm = [quantize_multiplier(float(sx[0]) * float(swv[c]) / float(sy[0])) for c in range(OC)]
            L["mult"] = np.array([a for a, _ in mm], np.int32)
            L["shift"] = np.array([b for _, b in mm], np.int32)
            L["act_min"], L["act_max"] = act_range(o.fusedActivationFunction, float(sy[0]), int(zy[0]))
        elif op == "FC":
            w_t = L["ins"][1]
            b_t = L["ins"][2] if len(L["ins"]) > 2 else -1
            ws = G.shape(w_t)
            sx, zx = G.q(x)
            sw_, zw = G.q(w_t)
            sy, zy = G.q(y)
            if np.any(zw != 0):
                die(f"layer {L['k']}: FC weight zero point must be 0")
            W = np.frombuffer(G.data(w_t), dtype=np.int8).reshape(ws)
            L["w"], L["wshape"] = W, ws
            OC = ws[0]
            swv = np.broadcast_to(sw_, (OC,)) if sw_.size == 1 else sw_
            L["bias"] = (np.frombuffer(G.data(b_t), dtype=np.int32).copy() if b_t >= 0 and G.data(b_t)
                         else np.zeros(OC, np.int32))
            mm = [quantize_multiplier(float(sx[0]) * float(swv[c]) / float(sy[0])) for c in range(OC)]
            L["mult"] = np.array([a for a, _ in mm], np.int32)
            L["shift"] = np.array([b for _, b in mm], np.int32)
            L["act_min"], L["act_max"] = act_range(o.fusedActivationFunction if o else 0, float(sy[0]), int(zy[0]))
        elif op in ("MAXPOOL", "AVGPOOL"):
            xi, yo = G.d4(x), G.d4(y)
            L.update(kh=o.filterHeight, kw=o.filterWidth, sh=o.strideH, sw=o.strideW)
            if o.padding == 0:
                L["pt"], L["pb"] = same_pad(xi[1], yo[1], o.filterHeight, o.strideH, 1)
                L["pl"], L["pr"] = same_pad(xi[2], yo[2], o.filterWidth, o.strideW, 1)
            sy, zy = G.q(y)
            L["act_min"], L["act_max"] = act_range(o.fusedActivationFunction, float(sy[0]), int(zy[0]))
        elif op == "PAD":
            pads = np.frombuffer(G.data(L["ins"][1]), dtype=np.int32).reshape(-1, 2)
            pads = np.vstack([np.zeros((4 - len(pads), 2), np.int32), pads])
            if pads[0].any() or pads[3].any():
                die(f"layer {L['k']}: only H/W padding supported")
            L.update(pt=int(pads[1][0]), pb=int(pads[1][1]), pl=int(pads[2][0]), pr=int(pads[2][1]))
            sy, zy = G.q(y)
            pv = int(zy[0])
            if L["name"] == "PADV2" and len(L["ins"]) > 2 and G.data(L["ins"][2]):
                pv = int(np.frombuffer(G.data(L["ins"][2]), dtype=np.int8)[0])
            L["i"][0] = pv
        elif op == "QUANTIZE":
            sy, zy = G.q(y)
            L["f"] = [float(sy[0]), float(np.float32(1.0) / np.float32(sy[0]))]
            L["i"][0] = int(zy[0])
            L["p"][2] = G.dtype(y)
            if G.dtype(x) != DT_F32:
                sx, zx = G.q(x)
                m_, s_ = quantize_multiplier(float(sx[0]) / float(sy[0]))
                L["p"][1] = 1
                L["i"][1] = int(zx[0])
                L["meanq"] = (m_, s_)
        elif op == "DEQUANTIZE":
            sx, zx = G.q(x)
            L["f"][0], L["i"][0] = float(sx[0]), int(zx[0])
        elif op == "LOGISTIC":
            sx, zx = G.q(x)
            sy, zy = G.q(y)
            lut = np.zeros(256, np.int8)
            for v in range(-128, 128):
                d = float(np.float32(sx[0]) * np.float32(v - int(zx[0])))
                tval = 1.0 / (1.0 + math.exp(-d))
                r = int(round_away(np.float32(tval) / np.float32(sy[0]))) + int(zy[0])
                lut[v + 128] = max(-128, min(127, r))
            L["lut"] = lut
        elif op == "SOFTMAX":
            sx, zx = G.q(x)
            sy, zy = G.q(y)
            beta = float(o.beta) if o is not None else 1.0
            sc = np.float32(-float(sx[0]) * beta)
            t16 = np.zeros(256, np.int64)
            for v in range(256):
                e = np.float32(np.exp(np.float32(sc * np.float32(255 - v))))
                t16[v] = min(65535, int(e * np.float32(65535.0) + np.float32(0.5)))
            L["lut16"] = t16
            L["f"] = [float(sy[0]), beta]
            L["i"] = [int(zy[0]), 0]
        elif op == "MEAN":
            axes = sorted(int(v) % 4 for v in np.frombuffer(G.data(L["ins"][1]), dtype=np.int32))
            if axes != [1, 2]:
                die(f"layer {L['k']}: MEAN only over H,W")
            sx, zx = G.q(x)
            sy, zy = G.q(y)
            xi = G.d4(x)
            n = xi[1] * xi[2]
            L["mean"] = dict(n=n, sx=float(sx[0]), zx=int(zx[0]), sy=float(sy[0]), zy=int(zy[0]))
            L["f"] = [float(sx[0]), float(sy[0])]
            L["i"] = [int(zx[0]), int(zy[0])]
        elif op == "RESHAPE":
            L["ins"] = [L["ins"][0]]
        elif op in ("ADD", "MUL"):
            a, b = L["ins"][0], L["ins"][1]
            if G.data(a) is not None and G.data(b) is not None:
                die(f"layer {L['k']}: {op} with two constant inputs not supported")
            if G.data(a) is not None:
                a, b = b, a
            L["in0"] = a
            ya = G.d4(y)
            for t in (a, b):
                for ax in range(1, 4):
                    if G.d4(t)[ax] not in (1, ya[ax]):
                        die(f"layer {L['k']}: {op} broadcast shape not supported")
            if G.d4(a) != ya:
                die(f"layer {L['k']}: {op} first input must have the output shape")
            if G.data(b) is not None:
                if G.dtype(b) != DT_I8:
                    die(f"layer {L['k']}: {op} constant must be int8")
                L["w"] = np.frombuffer(G.data(b), dtype=np.int8).copy()
                L["wshape"] = G.d4(b)
            else:
                L["in1"] = b
            L["bshape"] = G.d4(b)[1:4]
            s1, z1 = G.q(a)
            s2, z2 = G.q(b)
            sy, zy = G.q(y)
            if op == "ADD":
                tw = 2.0 * max(float(s1[0]), float(s2[0]))
                m1 = quantize_multiplier(float(s1[0]) / tw)
                m2 = quantize_multiplier(float(s2[0]) / tw)
                mo = quantize_multiplier(tw / ((1 << 20) * float(sy[0])))
            else:
                m1 = m2 = (0, 0)
                mo = quantize_multiplier(float(s1[0]) * float(s2[0]) / float(sy[0]))
            L["mult"] = np.array([m1[0], m2[0], mo[0]], np.int32)
            L["shift"] = np.array([m1[1], m2[1], mo[1]], np.int32)
            L["bias"] = np.zeros(3, np.int32)
            L["i"] = [int(z1[0]), int(z2[0])]
            L["p"][0] = int(zy[0])
            L["act_min"], L["act_max"] = act_range(o.fusedActivationFunction if o else 0, float(sy[0]), int(zy[0]))

    return G, plan, plan_path, layers, in_t, out_t


def reference_run(G, layers, x_in):
    tfl = G.path
    try:
        it = tf.lite.Interpreter(model_path=tfl, experimental_preserve_all_tensors=True,
                                 experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    except Exception:
        it = tf.lite.Interpreter(model_path=tfl, experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    it.set_tensor(d["index"], x_in.reshape(d["shape"]).astype(d["dtype"]))
    it.invoke()
    ref = {int(d["index"]): x_in.reshape(d["shape"]).astype(d["dtype"])}
    for L in layers:
        ref[L["out"]] = it.get_tensor(L["out"]).copy()
    out = it.get_tensor(it.get_output_details()[0]["index"]).copy()
    return ref, out


def conv_ref(L, x, G):
    xi = G.d4(L["in0"])
    yo = G.d4(L["out"])
    H, W, C = xi[1], xi[2], xi[3]
    OH, OW, OC = yo[1], yo[2], yo[3]
    sx, zx = G.q(L["in0"])
    sy, zy = G.q(L["out"])
    xs = x.reshape(H, W, C).astype(np.float64) - float(zx[0])
    xp = np.zeros((H + L["pt"] + L["pb"] + 64, W + L["pl"] + L["pr"] + 64, C))
    xp[L["pt"]:L["pt"] + H, L["pl"]:L["pl"] + W] = xs
    KH, KW = L["kh"], L["kw"]
    out = np.zeros((OH, OW, OC), np.int64)
    if L["op"] == "CONV":
        Wm = L["w"].reshape(OC, KH * KW * C).astype(np.float64).T
    else:
        M = L["p"][0]
        Wd = L["w"].reshape(KH, KW, OC).astype(np.float64)
    step = max(1, 2000000 // max(1, OW * KH * KW * C))
    for y0 in range(0, OH, step):
        y1 = min(OH, y0 + step)
        patches = np.empty((y1 - y0, OW, KH, KW, C))
        for ky in range(KH):
            for kx in range(KW):
                iy = np.arange(y0, y1) * L["sh"] + ky * L["dh"]
                ix = np.arange(OW) * L["sw"] + kx * L["dw"]
                patches[:, :, ky, kx, :] = xp[iy][:, ix]
        if L["op"] == "CONV":
            acc = patches.reshape(-1, KH * KW * C) @ Wm
            acc = acc.reshape(y1 - y0, OW, OC)
        else:
            pr = np.repeat(patches, M, axis=-1) if M > 1 else patches
            acc = np.einsum("yxhwc,hwc->yxc", pr, Wd)
        out[y0:y1] = np.rint(acc).astype(np.int64)
    out = out + L["bias"].astype(np.int64)
    r = mbqm(out, L["mult"], L["shift"], ROUND_UP[L["op"]]) + int(zy[0])
    return np.clip(r, L["act_min"], L["act_max"]).astype(np.int8)


def run_numpy(G, layers, x_float, ref, isolated=True, in_t=None):
    env = {in_t: x_float} if in_t is not None else {}
    rep = []
    ok_all = True
    for L in layers:
        op = L["op"]
        y = L["out"]
        x = ref.get(L["in0"]) if isolated and L["in0"] in ref else env.get(L["in0"])
        if op == "QUANTIZE":
            lo_, hi_ = (0, 255) if L["p"][2] == DT_U8 else (-128, 127)
            if L["p"][1] == 1:
                m_, s_ = L["meanq"]
                q = mbqm(x.reshape(-1).astype(np.int64) - L["i"][1], m_, s_) + L["i"][0]
            else:
                q = np.rint(np.float32(x_float) * L["f"][1]) + L["i"][0]
            r = np.clip(q, lo_, hi_).astype(np.uint8 if L["p"][2] == DT_U8 else np.int8)
        elif op in ("CONV", "DWCONV"):
            r = conv_ref(L, x, G)
        elif op == "FC":
            sx, zx = G.q(L["in0"])
            sy, zy = G.q(y)
            xv = x.reshape(-1).astype(np.float64) - float(zx[0])
            acc = np.rint(L["w"].astype(np.float64) @ xv).astype(np.int64) + L["bias"].astype(np.int64)
            r = np.clip(mbqm(acc, L["mult"], L["shift"], ROUND_UP["FC"]) + int(zy[0]), L["act_min"], L["act_max"]).astype(np.int8)
        elif op in ("MAXPOOL", "AVGPOOL"):
            xi, yo = G.d4(L["in0"]), G.d4(y)
            xa = x.reshape(xi[1], xi[2], xi[3]).astype(np.int32)
            r = np.zeros((yo[1], yo[2], yo[3]), np.int32)
            for oy in range(yo[1]):
                for ox in range(yo[2]):
                    y0, x0 = oy * L["sh"] - L["pt"], ox * L["sw"] - L["pl"]
                    win = xa[max(0, y0):min(xi[1], y0 + L["kh"]), max(0, x0):min(xi[2], x0 + L["kw"])]
                    if op == "MAXPOOL":
                        r[oy, ox] = win.max(axis=(0, 1))
                    else:
                        cnt = win.shape[0] * win.shape[1]
                        s = win.sum(axis=(0, 1))
                        r[oy, ox] = np.where(s > 0, (s + cnt // 2) // cnt, -((-s + cnt // 2) // cnt))
            r = np.clip(r, L["act_min"], L["act_max"]).astype(np.int8)
        elif op == "PAD":
            xi, yo = G.d4(L["in0"]), G.d4(y)
            r = np.full((yo[1], yo[2], yo[3]), L["i"][0], np.int8)
            r[L["pt"]:L["pt"] + xi[1], L["pl"]:L["pl"] + xi[2]] = x.reshape(xi[1], xi[2], xi[3])
        elif op == "RESHAPE":
            r = x.reshape(-1).copy()
        elif op in ("ADD", "MUL"):
            ys = G.d4(y)[1:4]
            if L["in1"] >= 0:
                x2 = ref.get(L["in1"]) if isolated and L["in1"] in ref else env.get(L["in1"])
            else:
                x2 = L["w"]
            xa = x.reshape(ys).astype(np.int64)
            xb = np.broadcast_to(np.asarray(x2).reshape(L["bshape"]).astype(np.int64), ys)
            up = ROUND_UP[op]
            if op == "ADD":
                a = (xa - L["i"][0]) << 20
                b = (xb - L["i"][1]) << 20
                sa = mbqm(a, int(L["mult"][0]), int(L["shift"][0]), up)
                sb = mbqm(b, int(L["mult"][1]), int(L["shift"][1]), up)
                r = mbqm(sa + sb, int(L["mult"][2]), int(L["shift"][2]), up) + L["p"][0]
            else:
                r = mbqm((xa - L["i"][0]) * (xb - L["i"][1]), int(L["mult"][2]), int(L["shift"][2]), up) + L["p"][0]
            r = np.clip(r, L["act_min"], L["act_max"]).astype(np.int8)
        elif op == "SOFTMAX":
            C = G.d4(L["in0"])[3]
            xs = x.reshape(-1, C).astype(np.int64) + 128
            out = np.zeros_like(xs)
            for row in range(xs.shape[0]):
                u = xs[row]
                idx = u + (255 - u.max())
                ev = L["lut16"][idx]
                ssum = int(ev.sum())
                inv = float(np.float32(1.0) / (np.float32(ssum) * np.float32(L["f"][0])))
                mlt, sh = quantize_multiplier(inv)
                out[row] = mbqm(ev, mlt, sh, ROUND_UP["SOFTMAX"]) + L["i"][0]
            r = np.clip(out, -128, 127).astype(np.int8)
        elif op == "MEAN":
            mp = L["mean"]
            s = x.reshape(-1, G.d4(L["in0"])[3]).astype(np.int64).sum(axis=0)
            mlt, sh = mean_multiplier(mp["sx"], mp["sy"], mp["n"])
            r = np.clip(mbqm(s - mp["n"] * mp["zx"], mlt, sh, ROUND_UP["MEAN"]) + mp["zy"], -128, 127).astype(np.int8)
        elif op == "LOGISTIC":
            r = L["lut"][x.reshape(-1).astype(np.int32) + 128]
        elif op == "DEQUANTIZE":
            r = (x.astype(np.float32) - np.float32(L["i"][0])) * np.float32(L["f"][0])
        else:
            r = ref[y]
        want = ref[y]
        r = np.asarray(r).reshape(want.shape).astype(want.dtype)
        if want.dtype == np.float32:
            diff = float(np.max(np.abs(r - want))) if want.size else 0.0
            same = diff <= 1e-6 * max(1.0, float(np.max(np.abs(want))))
            info = f"max diff {diff:.2e}"
        else:
            nd = int(np.sum(r != want))
            same = nd == 0
            info = f"{nd} values differ" if nd else "bit-exact"
        if op != "NOP":
            rep.append(f"  #{L['k']:<3} {L['name']:<18} {info}")
            ok_all &= bool(same)
        env[y] = r
    return ok_all, rep


G_in_lo = [0.0, 0.0]


def make_input(G, in_t, layers):
    n = int(np.prod(G.d4(in_t)))
    st = LCG_SEED
    v = np.empty(n, np.float32)
    for i in range(n):
        st = (st * 1664525 + 1013904223) & 0xFFFFFFFF
        v[i] = (st >> 8) * (1.0 / 16777216.0)
    if G.dtype(in_t) == DT_I8:
        return np.clip(np.floor(v * 256.0) - 128, -128, 127).astype(np.int8), -128, 127
    if G.dtype(in_t) == DT_U8:
        return np.clip(np.floor(v * 256.0), 0, 255).astype(np.uint8), 0, 255
    q0 = next((L for L in layers if L["op"] == "QUANTIZE" and L["in0"] == in_t), None)
    if q0 is None:
        die("float input without a QUANTIZE layer: the model must be int8 inside")
    s, z = q0["f"][0], q0["i"][0]
    lo, hi = (-128 - z) * s, (127 - z) * s
    return (np.float32(lo) + v * np.float32(hi - lo)).astype(np.float32), lo, hi


def plan_buffers(G, plan, layers, in_t, out_t):
    nL = len(layers)
    prod = {L["out"]: L["k"] for L in layers}
    cons = {}
    for L in layers:
        ts = [L["in0"], L["in1"]] if L["op"] != "NOP" else L["ins"][:1]
        for t in ts:
            if t >= 0:
                cons.setdefault(t, []).append(L["k"])
    if plan is None:
        groups = [dict(first=i, last=i, whole=True, rows={}, bufs={}) for i in range(nL)]
        spilled, part_off, inp = {}, {}, None
    else:
        groups = []
        for P in plan["groups"]:
            bufs = {}
            for b in P["buffers"]:
                bufs.setdefault(int(b["tensor"]), []).append((b["role"], int(b["rows"])))
            groups.append(dict(first=P["first_layer"], last=P["last_layer"], whole=bool(P["whole"]),
                               rows={int(k): int(v) for k, v in P["rows_per_step"].items()}, bufs=bufs))
        part_off = {p["name"]: int(p["offset"]) for p in plan["partitions"]}
        spilled = {int(s["tensor"]): part_off[s["partition"]] + int(s["offset"]) for s in plan["spilled"]}
        inp = plan["input"]
        spilled[in_t] = part_off[inp["location"]] + int(inp["offset"])
    gid = {}
    for k, Gp in enumerate(groups):
        for i in range(Gp["first"], Gp["last"] + 1):
            gid[i] = k

    def tbytes(t):
        return int(np.prod(G.d4(t))) * ESZ[G.dtype(t)]

    def rowb(t):
        d = G.d4(t)
        return d[2] * d[3] * ESZ[G.dtype(t)]

    insts = []
    use = {}

    def new_inst(t, sd, ext, g0, g1, cap):
        h = G.d4(t)[1]
        cap = h if cap is None else max(1, min(cap, h))
        insts.append(dict(t=t, sd=sd, ext=ext, g0=g0, g1=g1, cap=cap, size=cap * rowb(t)))
        return len(insts) - 1

    for t in sorted(set(list(prod) + list(cons) + [in_t])):
        pg = gid[prod[t]] if t in prod else -1
        cg = sorted(set(gid[c] for c in cons.get(t, [])))
        if t in spilled:
            if pg >= 0:
                cap = None
                if not groups[pg]["whole"]:
                    cap = next((r for role, r in groups[pg]["bufs"].get(t, []) if role == "out-sd"), None)
                i = new_inst(t, 1, spilled[t], pg, pg, cap)
                use[(t, pg)] = i
            for g in cg:
                if g == pg:
                    continue
                cap = None
                if not groups[g]["whole"]:
                    cap = next((r for role, r in groups[g]["bufs"].get(t, []) if role == "in-sd"), None)
                use[(t, g)] = new_inst(t, 2, spilled[t], g, g, cap)
        else:
            gs = ([pg] if pg >= 0 else []) + cg
            g0, g1 = (min(gs), max(gs)) if gs else (0, 0)
            if t == out_t:
                g1 = len(groups) - 1
            cap = None
            if g0 == g1 and not groups[g0]["whole"] and t != out_t:
                cap = next((r for role, r in groups[g0]["bufs"].get(t, []) if role == "int"), None)
            i = new_inst(t, 0, -1, g0, g1, cap)
            for g in set(gs):
                use[(t, g)] = i

    ALIGN = 8
    placed = []
    for idx in sorted(range(len(insts)), key=lambda i: -insts[i]["size"]):
        I = insts[idx]
        sz = (I["size"] + ALIGN - 1) // ALIGN * ALIGN
        conf = sorted((a, b) for (a, b, g0, g1) in placed if not (I["g1"] < g0 or g1 < I["g0"]))
        off = 0
        for a, b in conf:
            if off + sz <= a:
                break
            off = max(off, b)
        I["off"] = off
        placed.append((off, off + sz, I["g0"], I["g1"]))
    arena = max((p[1] for p in placed), default=8)
    return groups, gid, insts, use, arena


def emit(model, G, plan, layers, in_t, out_t, x_float, ref, out_final, rep, ok, outdir):
    os.makedirs(outdir, exist_ok=True)
    groups, gid, insts, use, arena = plan_buffers(G, plan, layers, in_t, out_t)

    if plan is not None:
        part = {p["name"]: p for p in plan["partitions"]}
        wpart = part.get("Weights")
        w_base = int(wpart["offset"]) if wpart else 0
        w_limit = int(wpart["size"]) if wpart else None
        in_off = int(part[plan["input"]["location"]]["offset"]) + int(plan["input"]["offset"])
    else:
        w_base, w_limit = 0, None
        in_off = None

    wbin = bytearray()
    for L in layers:
        if L["w"] is not None:
            pad = (-len(wbin)) % SECTOR
            wbin += b"\0" * pad
            L["w_off"] = w_base + len(wbin)
            L["w_bytes"] = L["w"].size
            wbin += L["w"].tobytes()
        else:
            L["w_off"], L["w_bytes"] = 0, 0
    wbin += b"\0" * ((-len(wbin)) % SECTOR)
    if WEIGHTS_IN_FLASH:
        w_base = 0
        with open(os.path.join(outdir, "nn_weights.h"), "w") as f:
            f.write("#ifndef NN_WEIGHTS_H\n#define NN_WEIGHTS_H\n#include <stdint.h>\n")
            f.write(f"#define NN_WEIGHTS_BYTES {len(wbin)}u\n")
            f.write("extern const int8_t nn_weights[NN_WEIGHTS_BYTES];\n#endif\n")
        arr8 = np.frombuffer(bytes(wbin), dtype=np.int8)
        with open(os.path.join(outdir, "nn_weights.c"), "w") as f:
            f.write('#include "nn_weights.h"\n\n__attribute__((aligned(4))) const int8_t nn_weights[NN_WEIGHTS_BYTES] = {\n')
            for i in range(0, len(arr8), 32):
                f.write("    " + ",".join(map(str, arr8[i:i + 32].tolist())) + ",\n")
            f.write("};\n")
        if len(wbin) > FLASH_BYTES:
            print(f"  WARNING: weights {len(wbin)} B are larger than the MCU flash ({FLASH_BYTES} B): the build will not fit")
    elif w_limit is not None and len(wbin) > w_limit:
        die(f"weights {len(wbin)} B do not fit the plan's Weights partition ({w_limit} B)")
    if in_off is None:
        in_off = w_base + len(wbin) + 1024 * 1024
        in_off -= in_off % SECTOR
    if not WEIGHTS_IN_FLASH:
        open(os.path.join(outdir, "sd_weights.bin"), "wb").write(bytes(wbin))

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
            m_, s_ = mean_multiplier(mp["sx"], mp["sy"], mp["n"])
            L["p"][1], L["p"][2] = 2, mp["n"]
            L["i"] = [mp["zx"], mp["zy"]]
            L["meanq"] = (m_, s_)

    h = []
    h.append(f"#ifndef NN_MODEL_H\n#define NN_MODEL_H\n#include <stdint.h>\n")
    h.append(f'#define NN_MODEL_NAME "{model}"')
    h.append(f"#define NN_N_LAYERS {len(layers)}\n#define NN_N_BUFS {len(insts)}\n#define NN_N_GROUPS {len(groups)}")
    h.append(f"#define NN_ARENA_BYTES {arena}u")
    h.append(f"#define NN_WEIGHTS_IN_FLASH {1 if WEIGHTS_IN_FLASH else 0}")
    h.append(f"#define NN_SD_WEIGHTS_OFFSET {w_base}u\n#define NN_SD_WEIGHTS_BYTES {len(wbin)}u")
    h.append(f"#define NN_SD_INPUT_OFFSET {in_off}u\n#define NN_SD_INPUT_BYTES {x_float.nbytes}u")
    h.append(f"#define NN_INPUT_SEED {LCG_SEED}u\n#define NN_INPUT_DTYPE {G.dtype(in_t)}")
    h.append(f"#define NN_INPUT_LO {float(np.float32(G_in_lo[0])).hex()}f\n#define NN_INPUT_SPAN {float(np.float32(G_in_lo[1])).hex()}f")
    h.append(f"#define NN_RAM_BUDGET {plan['ram_budget'] if plan else 0}u")
    h.append("\nenum { " + ", ".join(f"NN_OP_{n} = {i}" for i, n in enumerate(OPS)) + " };")
    h.append("enum { NN_DT_I8 = 0, NN_DT_F32 = 1, NN_DT_I32 = 2, NN_DT_U8 = 3 };")
    h.append("enum { NN_SD_NONE = 0, NN_SD_OUT = 1, NN_SD_IN = 2 };\n")
    h.append("typedef struct {\n    uint32_t off;\n    uint32_t row_bytes;\n    uint16_t h, w, c, cap;\n"
             "    int16_t prod;\n    uint8_t dtype, sd;\n    uint32_t sd_off;\n} nn_buf_t;\n")
    h.append("typedef struct {\n    uint8_t op;\n    int8_t step;\n    int16_t in0, in1, out;\n"
             "    int16_t kh, kw, sh, sw, dh, dw, pt, pl, pb, pr;\n    int16_t act_min, act_max;\n"
             "    int16_t p[3];\n    int16_t bh, bw, bc;\n    int32_t qp, lut;\n    uint32_t w_off, w_bytes;\n"
             "    int32_t i0, i1, m, s;\n    float f0, f1;\n} nn_layer_t;\n")
    h.append("typedef struct {\n    int16_t first, last;\n    uint8_t whole;\n} nn_group_t;\n")
    h.append("extern const nn_buf_t nn_bufs[NN_N_BUFS];\nextern const nn_layer_t nn_layers[NN_N_LAYERS];")
    h.append("extern const nn_group_t nn_groups[NN_N_GROUPS];\nextern const int32_t nn_bias[];")
    h.append("extern const int32_t nn_mult[];\nextern const int8_t nn_shift[];\nextern const int8_t nn_lut[];\nextern const uint16_t nn_lut16[];")
    h.append("extern const char *const nn_layer_names[NN_N_LAYERS];")
    h.append("extern const int64_t nn_ref_sum[NN_N_LAYERS];")
    h.append(f"#define NN_IN_BUF {use[(in_t, gid[0])]}\n#define NN_OUT_BUF {use[(out_t, gid[len(layers) - 1])]}")
    h.append("\n#endif\n")
    open(os.path.join(outdir, "nn_model.h"), "w").write("\n".join(h))

    for I in insts:
        I["prod"] = -1
    for L in layers:
        i = use[(L["out"], gid[L["k"]])]
        insts[i]["prod"] = L["k"]

    c = ['#include "nn_model.h"\n', "const nn_buf_t nn_bufs[NN_N_BUFS] = {"]
    for I in insts:
        d = G.d4(I["t"])
        rb = d[2] * d[3] * ESZ[G.dtype(I["t"])]
        c.append(f"    {{ {I['off']}u, {rb}u, {d[1]}, {d[2]}, {d[3]}, {I['cap']}, {I['prod']}, "
                 f"{G.dtype(I['t'])}, {I['sd']}, {max(I['ext'], 0)}u }},")
    c.append("};\n\nconst nn_layer_t nn_layers[NN_N_LAYERS] = {")
    for L in layers:
        g = gid[L["k"]]
        in0 = use.get((L["in0"], g), -1) if L["in0"] >= 0 else -1
        in1 = use.get((L["in1"], g), -1) if L["in1"] >= 0 else -1
        out = use[(L["out"], g)]
        step = groups[g]["rows"].get(L["k"], 1) if not groups[g]["whole"] else 0
        m_, s_ = L.get("meanq", (0, 0))
        c.append(f"    {{ NN_OP_{L['op']}, {step}, {in0}, {in1}, {out}, {L['kh']}, {L['kw']}, {L['sh']}, {L['sw']}, "
                 f"{L['dh']}, {L['dw']}, {L['pt']}, {L['pl']}, {L['pb']}, {L['pr']}, {L['act_min']}, {L['act_max']}, "
                 f"{{ {L['p'][0]}, {L['p'][1]}, {L['p'][2]} }}, {L['bshape'][0]}, {L['bshape'][1]}, {L['bshape'][2]}, {L['qp_idx']}, {L['lut_idx']}, {L['w_off']}u, {L['w_bytes']}u, "
                 f"{L['i'][0]}, {L['i'][1]}, {m_}, {s_}, {L['f'][0]!r}f, {L['f'][1]!r}f }},")
    c.append("};\n\nconst nn_group_t nn_groups[NN_N_GROUPS] = {")
    for Gp in groups:
        c.append(f"    {{ {Gp['first']}, {Gp['last']}, {1 if Gp['whole'] else 0} }},")
    c.append("};\n")

    def arr(name, typ, vals, per=12):
        vals = vals if vals else [0]
        c.append(f"const {typ} {name}[] = {{")
        for i in range(0, len(vals), per):
            c.append("    " + ", ".join(str(int(v)) for v in vals[i:i + per]) + ",")
        c.append("};\n")

    arr("nn_bias", "int32_t", bias_all, 8)
    arr("nn_mult", "int32_t", mult_all, 8)
    arr("nn_shift", "int8_t", shift_all, 16)
    arr("nn_lut", "int8_t", lut_all, 16)
    arr("nn_lut16", "uint16_t", lut16_all, 16)
    c.append("const char *const nn_layer_names[NN_N_LAYERS] = {")
    for L in layers:
        c.append(f'    "{L["name"]}",')
    c.append("};\n")
    sums = []
    for L in layers:
        v = ref[L["out"]]
        sums.append(int(np.sum(v.astype(np.int64))) if v.dtype in (np.int8, np.uint8) else 0)
    c.append("const int64_t nn_ref_sum[NN_N_LAYERS] = {")
    for i in range(0, len(sums), 6):
        c.append("    " + ", ".join(f"{v}LL" for v in sums[i:i + 6]) + ",")
    c.append("};")
    open(os.path.join(outdir, "nn_model.c"), "w").write("\n".join(c) + "\n")

    print(f"  arena {arena} B (budget {plan['ram_budget'] if plan else '-'} B), groups {len(groups)}")
    if WEIGHTS_IN_FLASH:
        print(f"  nn_weights.c   {len(wbin)} B of weights in MCU flash")
    else:
        print(f"  sd_weights.bin {len(wbin)} B -> SD sector {w_base // SECTOR}")
    print(f"  each layer vs TFLite: {'ALL LAYERS MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("\n".join(rep))
    print("  TFLite output:", " ".join(f"{v:.6f}" for v in out_final.reshape(-1)[:16]))

def run_one(tfl, outdir):
    name = os.path.splitext(os.path.basename(tfl))[0]
    G, plan, plan_path, layers, in_t, out_t = build(tfl)
    x_in, lo, hi = make_input(G, in_t, layers)
    G_in_lo[0], G_in_lo[1] = lo, hi - lo
    ref, out_final = reference_run(G, layers, x_in)
    ok, rep = run_numpy(G, layers, x_in, ref, isolated=True, in_t=in_t)
    ok2, _ = run_numpy(G, layers, x_in, ref, isolated=False, in_t=in_t)
    print(f"  full chain (input -> output) vs TFLite: {'ALL LAYERS MATCH' if ok2 else 'MISMATCH'}")
    emit(name, G, plan, layers, in_t, out_t, x_in, ref, out_final, rep, ok, outdir)
    print(f"  files written to {outdir}")


def main():
    jobs = dict(TFLITE_PATHS)
    args = sys.argv[1:]
    for i in range(0, len(args) - 1, 2):
        jobs[args[i]] = args[i + 1]
    if not jobs:
        print("usage: python3 tflite2c.py <model.tflite> <out_dir> [<model.tflite> <out_dir> ...]")
    for tfl, outdir in jobs.items():
        try:
            run_one(tfl, outdir)
        except SystemExit as e:
            print(f"  {e}")


main()
