#ifndef NN_PLACE_H
#define NN_PLACE_H
#include <stdint.h>
#include "nn_model.h"
#define NN_PLAN_SRAM_KB 256u
#define NN_SRAM_ARENA_BYTES 109952u
#define NN_SDRAM_EXTENT 0u
#define NN_ACT_SRAM_BYTES 359436u
#define NN_ACT_SDRAM_BYTES 0u
#define NN_N_SDRAM_BUFS 0u
enum { NN_POOL_SRAM = 0, NN_POOL_SDRAM = 1 };
typedef struct { uint8_t pool; uint32_t off; uint32_t bytes; } nn_place_t;
extern const nn_place_t nn_place[NN_N_BUFS];
extern const int16_t nn_in_zp[NN_N_LAYERS];
extern const int16_t nn_out_zp[NN_N_LAYERS];
#endif
