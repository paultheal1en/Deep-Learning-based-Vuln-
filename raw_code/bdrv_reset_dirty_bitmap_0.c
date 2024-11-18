void bdrv_reset_dirty_bitmap(BdrvDirtyBitmap *bitmap,

                             int64_t cur_sector, int64_t nr_sectors)

{

    assert(bdrv_dirty_bitmap_enabled(bitmap));

    hbitmap_reset(bitmap->bitmap, cur_sector, nr_sectors);

}
