int bdrv_is_encrypted(BlockDriverState *bs)

{

    if (bs->backing_hd && bs->backing_hd->encrypted)

        return 1;

    return bs->encrypted;

}
