static void raw_lock_medium(BlockDriverState *bs, bool locked)

{

    bdrv_lock_medium(bs->file->bs, locked);

}
