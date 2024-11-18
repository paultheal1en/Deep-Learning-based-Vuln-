static BlockDriverAIOCB *bdrv_aio_rw_vector(BlockDriverState *bs,

                                            int64_t sector_num,

                                            QEMUIOVector *qiov,

                                            int nb_sectors,

                                            BlockDriverCompletionFunc *cb,

                                            void *opaque,

                                            int is_write)



{

    BlockDriverAIOCBSync *acb;



    acb = qemu_aio_get(&bdrv_em_aiocb_info, bs, cb, opaque);

    acb->is_write = is_write;

    acb->qiov = qiov;

    acb->bounce = qemu_blockalign(bs, qiov->size);

    acb->bh = aio_bh_new(bdrv_get_aio_context(bs), bdrv_aio_bh_cb, acb);



    if (is_write) {

        qemu_iovec_to_buf(acb->qiov, 0, acb->bounce, qiov->size);

        acb->ret = bs->drv->bdrv_write(bs, sector_num, acb->bounce, nb_sectors);

    } else {

        acb->ret = bs->drv->bdrv_read(bs, sector_num, acb->bounce, nb_sectors);

    }



    qemu_bh_schedule(acb->bh);



    return &acb->common;

}
