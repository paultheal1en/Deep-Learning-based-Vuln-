static void bdrv_ioctl_bh_cb(void *opaque)

{

    BdrvIoctlCompletionData *data = opaque;



    bdrv_co_io_em_complete(data->co, -ENOTSUP);

    qemu_bh_delete(data->bh);

}
