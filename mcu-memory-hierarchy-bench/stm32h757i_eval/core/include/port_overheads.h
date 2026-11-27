#ifndef PORT_OVERHEADS_H
#define PORT_OVERHEADS_H

#include <stdint.h>

#define PORT_OV_FAIL 0xFFFFFFFFu

void *port_ov_buffer(uint32_t *bytes);
uint32_t port_ov_mpu_reprogram(void);
uint32_t port_ov_cache_clean_invalidate(void *p, uint32_t bytes);
void port_ov_cache_clean(void *p, uint32_t bytes);
void port_ov_cache_invalidate(void *p, uint32_t bytes);
int port_ov_dma_init(void);
uint32_t port_ov_dma_copy(const void *src, void *dst, uint32_t bytes);

#endif
