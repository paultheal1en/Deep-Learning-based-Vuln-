static int coroutine_fn bdrv_co_io_em(BlockDriverState *bs, int64_t sector_num,

                                      int nb_sectors, QEMUIOVector *iov,

                                      bool is_write)

{

    CoroutineIOCompletion co = {

        .coroutine = qemu_coroutine_self(),

    };

    BlockDriverAIOCB *acb;



    if (is_write) {

        acb = bdrv_aio_writev(bs, sector_num, iov, nb_sectors,

                              bdrv_co_io_em_complete, &co);

    } else {

        acb = bdrv_aio_readv(bs, sector_num, iov, nb_sectors,

                             bdrv_co_io_em_complete, &co);

    }



    trace_bdrv_co_io_em(bs, sector_num, nb_sectors, is_write, acb);

    if (!acb) {

        return -EIO;

    }

    qemu_coroutine_yield();



    return co.ret;

}
