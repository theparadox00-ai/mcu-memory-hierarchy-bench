#ifndef NN_ENGINE_H
#define NN_ENGINE_H
#include <stdint.h>
#include "nn_model.h"
#include "nn_place.h"

#define NN_SCRATCH_BYTES (8192u * 4u)

void *nn_buf_ptr(int b);
int nn_init(void);
void nn_load_input(void);
void nn_run(void);
int nn_verify(int verbose);
uint32_t nn_checksum(void);
int nn_argmax(void);

#endif
