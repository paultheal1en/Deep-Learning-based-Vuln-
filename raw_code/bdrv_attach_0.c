int bdrv_attach(BlockDriverState *bs, DeviceState *qdev)

{

    if (bs->peer) {

        return -EBUSY;

    }

    bs->peer = qdev;

    return 0;

}
