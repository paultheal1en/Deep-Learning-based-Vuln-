int bdrv_dirty_bitmap_get_meta(BlockDriverState *bs,

                               BdrvDirtyBitmap *bitmap, int64_t sector,

                               int nb_sectors)

{

    uint64_t i;

    int sectors_per_bit = 1 << hbitmap_granularity(bitmap->meta);



    /* To optimize: we can make hbitmap to internally check the range in a

     * coarse level, or at least do it word by word. */

    for (i = sector; i < sector + nb_sectors; i += sectors_per_bit) {

        if (hbitmap_get(bitmap->meta, i)) {

            return true;

        }

    }

    return false;

}
