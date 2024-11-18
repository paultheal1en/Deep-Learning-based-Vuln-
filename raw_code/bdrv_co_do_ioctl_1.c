static int bdrv_co_do_ioctl(BlockDriverState *bs, int req, void *buf)

{

    BlockDriver *drv = bs->drv;

    BdrvTrackedRequest tracked_req;

    CoroutineIOCompletion co = {

        .coroutine = qemu_coroutine_self(),

    };

    BlockAIOCB *acb;



    tracked_request_begin(&tracked_req, bs, 0, 0, BDRV_TRACKED_IOCTL);

    if (!drv || !drv->bdrv_aio_ioctl) {

        co.ret = -ENOTSUP;

        goto out;

    }



    acb = drv->bdrv_aio_ioctl(bs, req, buf, bdrv_co_io_em_complete, &co);

    if (!acb) {

        BdrvIoctlCompletionData *data = g_new(BdrvIoctlCompletionData, 1);

        data->bh = aio_bh_new(bdrv_get_aio_context(bs),

                                bdrv_ioctl_bh_cb, data);

        data->co = &co;

        qemu_bh_schedule(data->bh);

    }

    qemu_coroutine_yield();

out:

    tracked_request_end(&tracked_req);

    return co.ret;

}
