BlockDriverAIOCB *dma_bdrv_io(

    BlockDriverState *bs, QEMUSGList *sg, uint64_t sector_num,

    DMAIOFunc *io_func, BlockDriverCompletionFunc *cb,

    void *opaque, DMADirection dir)

{

    DMAAIOCB *dbs = qemu_aio_get(&dma_aiocb_info, bs, cb, opaque);



    trace_dma_bdrv_io(dbs, bs, sector_num, (dir == DMA_DIRECTION_TO_DEVICE));



    dbs->acb = NULL;

    dbs->bs = bs;

    dbs->sg = sg;

    dbs->sector_num = sector_num;

    dbs->sg_cur_index = 0;

    dbs->sg_cur_byte = 0;

    dbs->dir = dir;


    dbs->io_func = io_func;

    dbs->bh = NULL;

    qemu_iovec_init(&dbs->iov, sg->nsg);

    dma_bdrv_cb(dbs, 0);

    return &dbs->common;

}