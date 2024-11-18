int bdrv_is_snapshot(BlockDriverState *bs)

{

    return !!(bs->open_flags & BDRV_O_SNAPSHOT);

}
