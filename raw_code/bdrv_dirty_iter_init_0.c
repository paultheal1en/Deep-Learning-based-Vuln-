void bdrv_dirty_iter_init(BdrvDirtyBitmap *bitmap, HBitmapIter *hbi)

{

    hbitmap_iter_init(hbi, bitmap->bitmap, 0);

}
