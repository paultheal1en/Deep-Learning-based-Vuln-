int bdrv_debug_resume(BlockDriverState *bs, const char *tag)

{

    while (bs && (!bs->drv || !bs->drv->bdrv_debug_resume)) {

        bs = bs->file;

    }



    if (bs && bs->drv && bs->drv->bdrv_debug_resume) {

        return bs->drv->bdrv_debug_resume(bs, tag);

    }



    return -ENOTSUP;

}
