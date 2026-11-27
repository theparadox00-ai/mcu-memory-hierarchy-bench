#include "port.h"
#include "port_overheads.h"

void *port_ov_buffer(uint32_t *bytes)
{
    *bytes = 0u;
    return 0;
}

uint32_t port_ov_mpu_reprogram(void) { return PORT_OV_FAIL; }

uint32_t port_ov_cache_clean_invalidate(void *p, uint32_t bytes)
{
    (void)p;
    (void)bytes;
    return PORT_OV_FAIL;
}

void port_ov_cache_clean(void *p, uint32_t bytes)
{
    (void)p;
    (void)bytes;
}

void port_ov_cache_invalidate(void *p, uint32_t bytes)
{
    (void)p;
    (void)bytes;
}

int port_ov_dma_init(void) { return -1; }

uint32_t port_ov_dma_copy(const void *src, void *dst, uint32_t bytes)
{
    (void)src;
    (void)dst;
    (void)bytes;
    return PORT_OV_FAIL;
}
