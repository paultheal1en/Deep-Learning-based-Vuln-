void bdrv_dirty_bitmap_deserialize_zeroes(BdrvDirtyBitmap *bitmap,

                                          uint64_t start, uint64_t count,

                                          bool finish)

{

    hbitmap_deserialize_zeroes(bitmap->bitmap, start, count, finish);

}
