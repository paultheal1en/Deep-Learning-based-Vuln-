void bdrv_set_dirty_iter(HBitmapIter *hbi, int64_t offset)

{

    assert(hbi->hb);

    hbitmap_iter_init(hbi, hbi->hb, offset);

}
