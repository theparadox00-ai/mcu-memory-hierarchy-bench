#ifndef NN_MODEL_H
#define NN_MODEL_H
#include <stdint.h>

#define NN_MODEL_NAME "miniresnetv2_s2_64x50_tl_int8"
#define NN_N_LAYERS 36
#define NN_N_BUFS 37
#define NN_N_GROUPS 36
#define NN_ARENA_BYTES 109952u
#define NN_WEIGHTS_IN_FLASH 1
#define NN_SD_WEIGHTS_OFFSET 0u
#define NN_SD_WEIGHTS_BYTES 442880u
#define NN_SD_INPUT_OFFSET 1491456u
#define NN_SD_INPUT_BYTES 3200u
#define NN_INPUT_SEED 12345u
#define NN_INPUT_DTYPE 0
#define NN_INPUT_LO -0x1.0000000000000p+7f
#define NN_INPUT_SPAN 0x1.fe00000000000p+7f
#define NN_RAM_BUDGET 0u

enum { NN_OP_NOP = 0, NN_OP_QUANTIZE = 1, NN_OP_DEQUANTIZE = 2, NN_OP_CONV = 3, NN_OP_DWCONV = 4, NN_OP_FC = 5, NN_OP_MAXPOOL = 6, NN_OP_AVGPOOL = 7, NN_OP_PAD = 8, NN_OP_RESHAPE = 9, NN_OP_MEAN = 10, NN_OP_LOGISTIC = 11, NN_OP_ADD = 12, NN_OP_SOFTMAX = 13, NN_OP_MUL = 14 };
enum { NN_DT_I8 = 0, NN_DT_F32 = 1, NN_DT_I32 = 2, NN_DT_U8 = 3 };
enum { NN_SD_NONE = 0, NN_SD_OUT = 1, NN_SD_IN = 2 };

typedef struct {
    uint32_t off;
    uint32_t row_bytes;
    uint16_t h, w, c, cap;
    int16_t prod;
    uint8_t dtype, sd;
    uint32_t sd_off;
} nn_buf_t;

typedef struct {
    uint8_t op;
    int8_t step;
    int16_t in0, in1, out;
    int16_t kh, kw, sh, sw, dh, dw, pt, pl, pb, pr;
    int16_t act_min, act_max;
    int16_t p[3];
    int16_t bh, bw, bc;
    int32_t qp, lut;
    uint32_t w_off, w_bytes;
    int32_t i0, i1, m, s;
    float f0, f1;
} nn_layer_t;

typedef struct {
    int16_t first, last;
    uint8_t whole;
} nn_group_t;

extern const nn_buf_t nn_bufs[NN_N_BUFS];
extern const nn_layer_t nn_layers[NN_N_LAYERS];
extern const nn_group_t nn_groups[NN_N_GROUPS];
extern const int32_t nn_bias[];
extern const int32_t nn_mult[];
extern const int8_t nn_shift[];
extern const int8_t nn_lut[];
extern const uint16_t nn_lut16[];
extern const char *const nn_layer_names[NN_N_LAYERS];
extern const int64_t nn_ref_sum[NN_N_LAYERS];
#define NN_IN_BUF 0
#define NN_OUT_BUF 36

#endif
