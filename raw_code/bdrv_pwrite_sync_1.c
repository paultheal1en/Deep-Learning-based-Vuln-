int bdrv_pwrite_sync(BlockDriverState *bs, int64_t offset,

    const void *buf, int count)

{

    int ret;



    ret = bdrv_pwrite(bs, offset, buf, count);

    if (ret < 0) {

        return ret;

    }



    /* No flush needed for cache=writethrough, it uses O_DSYNC */

    if ((bs->open_flags & BDRV_O_CACHE_MASK) != 0) {

        bdrv_flush(bs);

    }



    return 0;

}
