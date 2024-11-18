static void bdrv_delete(BlockDriverState *bs)

{

    assert(!bs->dev);

    assert(!bs->job);

    assert(bdrv_op_blocker_is_empty(bs));

    assert(!bs->refcnt);

    assert(QLIST_EMPTY(&bs->dirty_bitmaps));



    bdrv_close(bs);



    /* remove from list, if necessary */

    bdrv_make_anon(bs);




    g_free(bs);

}