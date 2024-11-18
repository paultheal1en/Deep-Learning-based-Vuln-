void bdrv_dirty_bitmap_serialize_part(const BdrvDirtyBitmap *bitmap,

                                      uint8_t *buf, uint64_t start,

                                      uint64_t count)

{

    hbitmap_serialize_part(bitmap->bitmap, buf, start, count);

}
