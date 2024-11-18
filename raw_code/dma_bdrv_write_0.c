BlockAIOCB *dma_bdrv_write(BlockDriverState *bs,

                           QEMUSGList *sg, uint64_t sector,

                           void (*cb)(void *opaque, int ret), void *opaque)

{

    return dma_bdrv_io(bs, sg, sector, bdrv_aio_writev, cb, opaque,

                       DMA_DIRECTION_TO_DEVICE);

}
