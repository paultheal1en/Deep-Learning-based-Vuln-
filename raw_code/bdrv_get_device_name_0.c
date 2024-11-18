const char *bdrv_get_device_name(const BlockDriverState *bs)

{

    return bs->blk ? blk_name(bs->blk) : "";

}
