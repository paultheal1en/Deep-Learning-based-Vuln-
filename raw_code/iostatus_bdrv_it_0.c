static void iostatus_bdrv_it(void *opaque, BlockDriverState *bs)

{

    bdrv_iostatus_reset(bs);

}
