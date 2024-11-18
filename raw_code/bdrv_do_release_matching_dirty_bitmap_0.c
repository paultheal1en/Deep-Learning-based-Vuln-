static void bdrv_do_release_matching_dirty_bitmap(BlockDriverState *bs,

                                                  BdrvDirtyBitmap *bitmap,

                                                  bool only_named)

{

    BdrvDirtyBitmap *bm, *next;

    QLIST_FOREACH_SAFE(bm, &bs->dirty_bitmaps, list, next) {

        if ((!bitmap || bm == bitmap) && (!only_named || bm->name)) {

            assert(!bm->active_iterators);

            assert(!bdrv_dirty_bitmap_frozen(bm));

            assert(!bm->meta);

            QLIST_REMOVE(bm, list);

            hbitmap_free(bm->bitmap);

            g_free(bm->name);

            g_free(bm);



            if (bitmap) {

                return;

            }

        }

    }

    if (bitmap) {

        abort();

    }

}
