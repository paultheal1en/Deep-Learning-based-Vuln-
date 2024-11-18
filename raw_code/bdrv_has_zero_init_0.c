int bdrv_has_zero_init(BlockDriverState *bs)

{

    assert(bs->drv);



    if (bs->drv->no_zero_init) {

        return 0;

    } else if (bs->file) {

        return bdrv_has_zero_init(bs->file);

    }



    return 1;

}
