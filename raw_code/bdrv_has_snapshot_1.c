static int bdrv_has_snapshot(BlockDriverState *bs)

{

    return (bs &&

            !bdrv_is_removable(bs) &&

            !bdrv_is_read_only(bs));

}
