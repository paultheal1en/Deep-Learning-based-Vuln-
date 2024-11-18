static int raw_has_zero_init(BlockDriverState *bs)

{

    return bdrv_has_zero_init(bs->file->bs);

}
