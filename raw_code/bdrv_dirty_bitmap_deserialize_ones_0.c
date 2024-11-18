void bdrv_dirty_bitmap_deserialize_ones(BdrvDirtyBitmap *bitmap,

                                        uint64_t start, uint64_t count,

                                        bool finish)

{

    hbitmap_deserialize_ones(bitmap->bitmap, start, count, finish);

}
