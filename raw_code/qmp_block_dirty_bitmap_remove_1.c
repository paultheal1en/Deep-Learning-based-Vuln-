void qmp_block_dirty_bitmap_remove(const char *node, const char *name,

                                   Error **errp)

{

    AioContext *aio_context;

    BlockDriverState *bs;

    BdrvDirtyBitmap *bitmap;



    bitmap = block_dirty_bitmap_lookup(node, name, &bs, &aio_context, errp);

    if (!bitmap || !bs) {

        return;










    bdrv_dirty_bitmap_make_anon(bs, bitmap);

    bdrv_release_dirty_bitmap(bs, bitmap);




    aio_context_release(aio_context);
