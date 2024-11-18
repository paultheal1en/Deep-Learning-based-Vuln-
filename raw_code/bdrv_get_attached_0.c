DeviceState *bdrv_get_attached(BlockDriverState *bs)

{

    return bs->peer;

}
