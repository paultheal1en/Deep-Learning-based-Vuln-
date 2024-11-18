DriveInfo *drive_get_by_blockdev(BlockDriverState *bs)

{

    return bs->blk ? blk_legacy_dinfo(bs->blk) : NULL;

}
