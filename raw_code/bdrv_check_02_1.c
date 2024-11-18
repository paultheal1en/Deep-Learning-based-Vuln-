int bdrv_check(BlockDriverState *bs)

{

    if (bs->drv->bdrv_check == NULL) {

        return -ENOTSUP;

    }



    return bs->drv->bdrv_check(bs);

}
