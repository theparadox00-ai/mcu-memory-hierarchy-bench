#include "nn_engine.h"
#include "nn_weights.h"
#include <string.h>
#include <stdio.h>
#include "port.h"

__attribute__((aligned(32))) static int8_t sram_arena[NN_SRAM_ARENA_BYTES];

static int8_t *ext_base;
static int32_t *SCRATCH;
#define SDRAM_PTR ext_base

#define WSUM_MAX 6144
#define ACC_OFF  WSUM_MAX

static int32_t wsum_off[NN_N_LAYERS];

void *nn_buf_ptr(int b)
{
    const nn_place_t *p = &nn_place[b];
    return p->pool == NN_POOL_SDRAM ? (void *)(SDRAM_PTR + p->off) : (void *)(sram_arena + p->off);
}

static inline int32_t mbqm(int32_t acc, int32_t m, int32_t s, int up)
{
    int32_t left = s > 0 ? s : 0;
    int32_t right = s > 0 ? 0 : -s;
    int64_t ab = ((int64_t)acc << left) * (int64_t)m;
    int64_t t = ab + (ab >= 0 ? (1ll << 30) : (1ll - (1ll << 30)));
    int64_t hi = t >= 0 ? (t >> 31) : -((-t) >> 31);
    if (hi > 2147483647ll) hi = 2147483647ll;
    if (hi < -2147483648ll) hi = -2147483648ll;
    if (right == 0) return (int32_t)hi;
    int64_t mask = (1ll << right) - 1;
    int64_t rem = hi & mask;
    int64_t thr = (mask >> 1) + ((!up && hi < 0) ? 1 : 0);
    return (int32_t)((hi >> right) + (rem > thr ? 1 : 0));
}

static inline int32_t clampi(int32_t v, int32_t lo, int32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

#if defined(__ARM_FEATURE_DSP)
static inline uint32_t sx16(uint32_t x) { uint32_t r; __asm("sxtb16 %0, %1" : "=r"(r) : "r"(x)); return r; }
static inline uint32_t sx16r(uint32_t x) { uint32_t r; __asm("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(x)); return r; }
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t c) { int32_t r; __asm("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
#else
static inline uint32_t sx16(uint32_t x) { return ((uint32_t)(uint16_t)(int16_t)(int8_t)(x)) | ((uint32_t)(uint16_t)(int16_t)(int8_t)(x >> 16) << 16); }
static inline uint32_t sx16r(uint32_t x) { return sx16((x >> 8) | (x << 24)); }
static inline int32_t smlad(uint32_t a, uint32_t b, int32_t c) { return c + (int16_t)a * (int16_t)b + (int16_t)(a >> 16) * (int16_t)(b >> 16); }
#endif

static inline uint32_t ld32(const int8_t *p)
{
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

static inline int32_t dot(const int8_t *x, const int8_t *w, int n)
{
    int32_t acc = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t xv = ld32(x + i), wv = ld32(w + i);
        acc = smlad(sx16(xv), sx16(wv), acc);
        acc = smlad(sx16r(xv), sx16r(wv), acc);
    }
    for (; i < n; i++) acc += (int32_t)x[i] * w[i];
    return acc;
}

static inline void dot2(const int8_t *x, const int8_t *w0, const int8_t *w1, int n, int32_t *a0, int32_t *a1)
{
    int32_t s0 = 0, s1 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t xv = ld32(x + i);
        uint32_t xe = sx16(xv), xo = sx16r(xv);
        uint32_t v0 = ld32(w0 + i), v1 = ld32(w1 + i);
        s0 = smlad(xe, sx16(v0), s0);
        s0 = smlad(xo, sx16r(v0), s0);
        s1 = smlad(xe, sx16(v1), s1);
        s1 = smlad(xo, sx16r(v1), s1);
    }
    for (; i < n; i++) { s0 += (int32_t)x[i] * w0[i]; s1 += (int32_t)x[i] * w1[i]; }
    *a0 = s0;
    *a1 = s1;
}

int nn_init(void)
{
    uint32_t sb = 0u;
    ext_base = (int8_t *)port_extmem_base();
    SCRATCH = (int32_t *)port_scratch(&sb);
    if (sb < NN_SCRATCH_BYTES) return -1;
    int pos = 0;
    for (int k = 0; k < NN_N_LAYERS; k++) {
        const nn_layer_t *L = &nn_layers[k];
        wsum_off[k] = -1;
        if (L->op != NN_OP_CONV && L->op != NN_OP_FC) continue;
        const nn_buf_t *X = &nn_bufs[L->in0];
        const nn_buf_t *Y = &nn_bufs[L->out];
        int oc = Y->c;
        int per = (L->op == NN_OP_FC) ? (int)(L->w_bytes / (uint32_t)oc) : L->kh * L->kw * X->c;
        if (pos + oc > WSUM_MAX) { wsum_off[k] = -2; continue; }
        wsum_off[k] = pos;
        for (int o = 0; o < oc; o++) {
            const int8_t *w = nn_weights + L->w_off + (uint32_t)o * per;
            int32_t s = 0;
            for (int i = 0; i < per; i++) s += w[i];
            SCRATCH[pos + o] = s;
        }
        pos += oc;
    }
    return 0;
}

static void op_quantize(const nn_layer_t *L, const nn_buf_t *X, const nn_buf_t *Y, const void *xp, void *yp)
{
    uint32_t n = (uint32_t)X->h * X->w * X->c;
    int lo = Y->dtype == NN_DT_U8 ? 0 : -128, hi = Y->dtype == NN_DT_U8 ? 255 : 127;
    if (L->p[1] == 1) {
        for (uint32_t i = 0; i < n; i++) {
            int32_t xv = X->dtype == NN_DT_U8 ? (int32_t)((const uint8_t *)xp)[i] : (int32_t)((const int8_t *)xp)[i];
            int32_t v = clampi(mbqm(xv - L->i1, L->m, L->s, 0) + L->i0, lo, hi);
            if (Y->dtype == NN_DT_U8) ((uint8_t *)yp)[i] = (uint8_t)v; else ((int8_t *)yp)[i] = (int8_t)v;
        }
    } else {
        const float *xf = (const float *)xp;
        for (uint32_t i = 0; i < n; i++) {
            float q = __builtin_rintf(xf[i] * L->f1);
            int32_t v = clampi((int32_t)q + L->i0, lo, hi);
            if (Y->dtype == NN_DT_U8) ((uint8_t *)yp)[i] = (uint8_t)v; else ((int8_t *)yp)[i] = (int8_t)v;
        }
    }
}

static void op_conv(int k, const nn_layer_t *L, const nn_buf_t *X, const nn_buf_t *Y, const int8_t *x, int8_t *y)
{
    const int H = X->h, W = X->w, C = X->c, OH = Y->h, OW = Y->w, OC = Y->c;
    const int zx = nn_in_zp[k];
    const int8_t *wt = nn_weights + L->w_off;
    const int32_t *bias = nn_bias + L->qp, *mult = nn_mult + L->qp;
    const int8_t *shf = nn_shift + L->qp;
    const int amin = L->act_min, amax = L->act_max;
    const int32_t *ws = wsum_off[k] >= 0 ? &SCRATCH[wsum_off[k]] : 0;
    if (L->kh == 1 && L->kw == 1 && L->sh == 1 && L->sw == 1 && L->pt == 0 && L->pl == 0 && ws) {
        const int npix = OH * OW;
        for (int p = 0; p < npix; p++) {
            const int8_t *xp = x + (size_t)p * C;
            int8_t *yp = y + (size_t)p * OC;
            int o = 0;
            for (; o + 2 <= OC; o += 2) {
                int32_t a0, a1;
                dot2(xp, wt + (size_t)o * C, wt + (size_t)(o + 1) * C, C, &a0, &a1);
                a0 = a0 - zx * ws[o] + bias[o];
                a1 = a1 - zx * ws[o + 1] + bias[o + 1];
                yp[o] = (int8_t)clampi(mbqm(a0, mult[o], shf[o], 1) + nn_out_zp[k], amin, amax);
                yp[o + 1] = (int8_t)clampi(mbqm(a1, mult[o + 1], shf[o + 1], 1) + nn_out_zp[k], amin, amax);
            }
            for (; o < OC; o++) {
                int32_t a = dot(xp, wt + (size_t)o * C, C) - zx * ws[o] + bias[o];
                yp[o] = (int8_t)clampi(mbqm(a, mult[o], shf[o], 1) + nn_out_zp[k], amin, amax);
            }
        }
        return;
    }
    const int KH = L->kh, KW = L->kw;
    for (int oy = 0; oy < OH; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            int8_t *yp = y + ((size_t)oy * OW + ox) * OC;
            int iy0 = oy * L->sh - L->pt, ix0 = ox * L->sw - L->pl;
            for (int o = 0; o < OC; o++) {
                const int8_t *wo = wt + (size_t)o * KH * KW * C;
                int32_t a = bias[o];
                for (int ky = 0; ky < KH; ky++) {
                    int iy = iy0 + ky * L->dh;
                    if (iy < 0 || iy >= H) continue;
                    for (int kx = 0; kx < KW; kx++) {
                        int ix = ix0 + kx * L->dw;
                        if (ix < 0 || ix >= W) continue;
                        const int8_t *xp = x + ((size_t)iy * W + ix) * C;
                        const int8_t *wp = wo + (ky * KW + kx) * C;
                        for (int c = 0; c < C; c++) a += ((int32_t)xp[c] - zx) * wp[c];
                    }
                }
                yp[o] = (int8_t)clampi(mbqm(a, mult[o], shf[o], 1) + nn_out_zp[k], amin, amax);
            }
        }
    }
}

static void op_dwconv(int k, const nn_layer_t *L, const nn_buf_t *X, const nn_buf_t *Y, const int8_t *x, int8_t *y)
{
    const int H = X->h, W = X->w, C = X->c, OH = Y->h, OW = Y->w, OC = Y->c, M = L->p[0];
    const int zx = nn_in_zp[k];
    const int8_t *wt = nn_weights + L->w_off;
    const int32_t *bias = nn_bias + L->qp, *mult = nn_mult + L->qp;
    const int8_t *shf = nn_shift + L->qp;
    int32_t *acc = &SCRATCH[ACC_OFF];
    const int KH = L->kh, KW = L->kw;
    for (int oy = 0; oy < OH; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            for (int o = 0; o < OC; o++) acc[o] = bias[o];
            int iy0 = oy * L->sh - L->pt, ix0 = ox * L->sw - L->pl;
            for (int ky = 0; ky < KH; ky++) {
                int iy = iy0 + ky * L->dh;
                if (iy < 0 || iy >= H) continue;
                for (int kx = 0; kx < KW; kx++) {
                    int ix = ix0 + kx * L->dw;
                    if (ix < 0 || ix >= W) continue;
                    const int8_t *xp = x + ((size_t)iy * W + ix) * C;
                    const int8_t *wp = wt + (ky * KW + kx) * OC;
                    if (M == 1) {
                        for (int c = 0; c < C; c++) acc[c] += ((int32_t)xp[c] - zx) * wp[c];
                    } else {
                        for (int c = 0; c < C; c++) {
                            int32_t xv = (int32_t)xp[c] - zx;
                            for (int m = 0; m < M; m++) acc[c * M + m] += xv * wp[c * M + m];
                        }
                    }
                }
            }
            int8_t *yp = y + ((size_t)oy * OW + ox) * OC;
            for (int o = 0; o < OC; o++)
                yp[o] = (int8_t)clampi(mbqm(acc[o], mult[o], shf[o], 1) + nn_out_zp[k], L->act_min, L->act_max);
        }
    }
}

static void op_fc(int k, const nn_layer_t *L, const nn_buf_t *X, const nn_buf_t *Y, const int8_t *x, int8_t *y)
{
    const int OC = Y->c;
    const int N = (int)(L->w_bytes / (uint32_t)OC);
    const int zx = nn_in_zp[k];
    const int8_t *wt = nn_weights + L->w_off;
    const int32_t *bias = nn_bias + L->qp, *mult = nn_mult + L->qp;
    const int8_t *shf = nn_shift + L->qp;
    (void)X;
    for (int o = 0; o < OC; o++) {
        const int8_t *w = wt + (size_t)o * N;
        int32_t s = 0, a = dot(x, w, N);
        if (wsum_off[k] >= 0) s = SCRATCH[wsum_off[k] + o];
        else for (int i = 0; i < N; i++) s += w[i];
        a = a - zx * s + bias[o];
        y[o] = (int8_t)clampi(mbqm(a, mult[o], shf[o], 1) + nn_out_zp[k], L->act_min, L->act_max);
    }
}

static void op_addmul(const nn_layer_t *L, const nn_buf_t *Y, const int8_t *a, const int8_t *b)
{
    int8_t *y = (int8_t *)nn_buf_ptr(L->out);
    const int H = Y->h, W = Y->w, C = Y->c;
    const int32_t m0 = nn_mult[L->qp], m1 = nn_mult[L->qp + 1], m2 = nn_mult[L->qp + 2];
    const int32_t s0 = nn_shift[L->qp], s1 = nn_shift[L->qp + 1], s2 = nn_shift[L->qp + 2];
    const int full = (L->bh == H && L->bw == W && L->bc == C);
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            size_t pa = ((size_t)yy * W + xx) * C;
            size_t pb = full ? pa : ((size_t)(L->bh == 1 ? 0 : yy) * L->bw + (L->bw == 1 ? 0 : xx)) * L->bc;
            for (int c = 0; c < C; c++) {
                int32_t xa = (int32_t)a[pa + c] - L->i0;
                int32_t xb = (int32_t)b[pb + (L->bc == 1 ? 0 : c)] - L->i1;
                int32_t r;
                if (L->op == NN_OP_ADD) {
                    int32_t sa = mbqm(xa * (1 << 20), m0, s0, 0);
                    int32_t sb = mbqm(xb * (1 << 20), m1, s1, 0);
                    r = mbqm(sa + sb, m2, s2, 0) + L->p[0];
                } else {
                    r = mbqm(xa * xb, m2, s2, 0) + L->p[0];
                }
                y[pa + c] = (int8_t)clampi(r, L->act_min, L->act_max);
            }
        }
    }
}

static void op_mean(const nn_layer_t *L, const nn_buf_t *X, const int8_t *x, int8_t *y)
{
    const int C = X->c, n = X->h * X->w;
    int32_t *acc = &SCRATCH[ACC_OFF];
    for (int c = 0; c < C; c++) acc[c] = 0;
    for (int p = 0; p < n; p++) {
        const int8_t *xp = x + (size_t)p * C;
        for (int c = 0; c < C; c++) acc[c] += xp[c];
    }
    for (int c = 0; c < C; c++)
        y[c] = (int8_t)clampi(mbqm(acc[c] - n * L->i0, L->m, L->s, 0) + L->i1, -128, 127);
}

static void qm(double d, int32_t *m, int32_t *s)
{
    if (d == 0.0) { *m = 0; *s = 0; return; }
    int e;
    double q = __builtin_frexp(d, &e);
    int64_t qf = (int64_t)__builtin_llround(q * 2147483648.0);
    if (qf == (1ll << 31)) { qf /= 2; e++; }
    if (e < -31) { *m = 0; *s = 0; return; }
    if (e > 30) { e = 30; qf = (1ll << 31) - 1; }
    *m = (int32_t)qf;
    *s = e;
}

static void op_softmax(const nn_layer_t *L, const nn_buf_t *X, const int8_t *x, int8_t *y)
{
    const int C = X->c;
    const int rows = (int)((uint32_t)X->h * X->w);
    const uint16_t *lut = nn_lut16 + L->lut;
    for (int r = 0; r < rows; r++) {
        const int8_t *xr = x + (size_t)r * C;
        int8_t *yr = y + (size_t)r * C;
        int mx = 0;
        for (int c = 0; c < C; c++) if (xr[c] + 128 > mx) mx = xr[c] + 128;
        int32_t sum = 0;
        for (int c = 0; c < C; c++) sum += lut[xr[c] + 128 + (255 - mx)];
        float inv = 1.0f / ((float)sum * L->f0);
        int32_t m, s;
        qm((double)inv, &m, &s);
        for (int c = 0; c < C; c++)
            yr[c] = (int8_t)clampi(mbqm(lut[xr[c] + 128 + (255 - mx)], m, s, 0) + L->i0, -128, 127);
    }
}

static void op_pad(const nn_layer_t *L, const nn_buf_t *X, const nn_buf_t *Y, const int8_t *x, int8_t *y)
{
    const int H = X->h, W = X->w, C = X->c, OW = Y->w;
    memset(y, (int8_t)L->i0, (size_t)Y->h * Y->w * Y->c);
    for (int r = 0; r < H; r++)
        memcpy(y + ((size_t)(r + L->pt) * OW + L->pl) * C, x + (size_t)r * W * C, (size_t)W * C);
}

static void op_maxpool(const nn_layer_t *L, const nn_buf_t *X, const nn_buf_t *Y, const int8_t *x, int8_t *y)
{
    const int H = X->h, W = X->w, C = X->c, OH = Y->h, OW = Y->w;
    for (int oy = 0; oy < OH; oy++) {
        for (int ox = 0; ox < OW; ox++) {
            int8_t *yp = y + ((size_t)oy * OW + ox) * C;
            int y0 = oy * L->sh - L->pt, x0 = ox * L->sw - L->pl;
            int ya = y0 < 0 ? 0 : y0, yb = y0 + L->kh > H ? H : y0 + L->kh;
            int xa = x0 < 0 ? 0 : x0, xb = x0 + L->kw > W ? W : x0 + L->kw;
            for (int c = 0; c < C; c++) {
                int m = -128;
                for (int iy = ya; iy < yb; iy++)
                    for (int ix = xa; ix < xb; ix++) {
                        int v = x[((size_t)iy * W + ix) * C + c];
                        if (v > m) m = v;
                    }
                yp[c] = (int8_t)clampi(m, L->act_min, L->act_max);
            }
        }
    }
}

static void run_layer(int k)
{
    const nn_layer_t *L = &nn_layers[k];
    const nn_buf_t *X = L->in0 >= 0 ? &nn_bufs[L->in0] : 0;
    const nn_buf_t *Y = &nn_bufs[L->out];
    void *xp = L->in0 >= 0 ? nn_buf_ptr(L->in0) : 0;
    void *yp = nn_buf_ptr(L->out);
    switch (L->op) {
    case NN_OP_QUANTIZE: op_quantize(L, X, Y, xp, yp); break;
    case NN_OP_CONV: op_conv(k, L, X, Y, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_DWCONV: op_dwconv(k, L, X, Y, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_FC: op_fc(k, L, X, Y, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_ADD:
    case NN_OP_MUL: {
        const int8_t *b = L->in1 >= 0 ? (const int8_t *)nn_buf_ptr(L->in1) : nn_weights + L->w_off;
        op_addmul(L, Y, (const int8_t *)xp, b);
        break;
    }
    case NN_OP_PAD: op_pad(L, X, Y, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_MAXPOOL: op_maxpool(L, X, Y, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_MEAN: op_mean(L, X, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_SOFTMAX: op_softmax(L, X, (const int8_t *)xp, (int8_t *)yp); break;
    case NN_OP_DEQUANTIZE: {
        uint32_t n = (uint32_t)X->h * X->w * X->c;
        const int8_t *xi = (const int8_t *)xp;
        float *yf = (float *)yp;
        for (uint32_t i = 0; i < n; i++) yf[i] = ((float)xi[i] - (float)L->i0) * L->f0;
        break;
    }
    case NN_OP_RESHAPE: {
        uint32_t n = nn_place[L->out].bytes;
        if (xp != yp) memmove(yp, xp, n);
        break;
    }
    default: break;
    }
}

void nn_load_input(void)
{
    uint32_t st = NN_INPUT_SEED;
    const nn_buf_t *B = &nn_bufs[NN_IN_BUF];
    uint32_t n = (uint32_t)B->h * B->w * B->c;
    uint8_t *p = (uint8_t *)nn_buf_ptr(NN_IN_BUF);
    for (uint32_t i = 0; i < n; i++) {
        st = st * 1664525u + 1013904223u;
        p[i] = (uint8_t)(B->dtype == NN_DT_U8 ? (st >> 24) : ((st >> 24) - 128u));
    }
}

void nn_run(void)
{
    for (int k = 0; k < NN_N_LAYERS; k++) run_layer(k);
}

static int64_t buf_sum(int b)
{
    const nn_buf_t *B = &nn_bufs[b];
    uint32_t n = (uint32_t)B->h * B->w * B->c;
    int64_t s = 0;
    if (B->dtype == NN_DT_U8) { const uint8_t *p = (const uint8_t *)nn_buf_ptr(b); for (uint32_t i = 0; i < n; i++) s += p[i]; }
    else if (B->dtype == NN_DT_I8) { const int8_t *p = (const int8_t *)nn_buf_ptr(b); for (uint32_t i = 0; i < n; i++) s += p[i]; }
    return s;
}

int nn_verify(int verbose)
{
    int bad = 0;
    for (int k = 0; k < NN_N_LAYERS; k++) {
        run_layer(k);
        int64_t s = buf_sum(nn_layers[k].out);
        if (s != nn_ref_sum[k]) {
            bad++;
            if (verbose) printf("  layer %d %s: sum %ld, TFLite %ld\n", k, nn_layer_names[k], (long)s, (long)nn_ref_sum[k]);
        }
    }
    return bad;
}

uint32_t nn_checksum(void)
{
    const nn_buf_t *B = &nn_bufs[NN_OUT_BUF];
    uint32_t n = (uint32_t)B->h * B->w * B->c * (B->dtype == NN_DT_F32 || B->dtype == NN_DT_I32 ? 4u : 1u);
    const uint8_t *p = (const uint8_t *)nn_buf_ptr(NN_OUT_BUF);
    uint32_t h = 2166136261u;
    for (uint32_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

int nn_argmax(void)
{
    const nn_buf_t *B = &nn_bufs[NN_OUT_BUF];
    int n = B->h * B->w * B->c, best = 0;
    if (B->dtype == NN_DT_F32) {
        const float *o = (const float *)nn_buf_ptr(NN_OUT_BUF);
        for (int i = 1; i < n; i++) if (o[i] > o[best]) best = i;
    } else {
        const int8_t *o = (const int8_t *)nn_buf_ptr(NN_OUT_BUF);
        for (int i = 1; i < n; i++) if (o[i] > o[best]) best = i;
    }
    return best;
}
