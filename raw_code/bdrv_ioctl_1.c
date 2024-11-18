int bdrv_ioctl(BlockDriverState *bs, unsigned long int req, void *buf)

{

    BdrvIoctlCoData data = {

        .bs = bs,

        .req = req,

        .buf = buf,

        .ret = -EINPROGRESS,

    };



    if (qemu_in_coroutine()) {

        /* Fast-path if already in coroutine context */

        bdrv_co_ioctl_entry(&data);

    } else {

        Coroutine *co = qemu_coroutine_create(bdrv_co_ioctl_entry);



        qemu_coroutine_enter(co, &data);

        while (data.ret == -EINPROGRESS) {

            aio_poll(bdrv_get_aio_context(bs), true);

        }

    }

    return data.ret;

}
