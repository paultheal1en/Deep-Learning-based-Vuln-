int bdrv_all_delete_snapshot(const char *name, BlockDriverState **first_bad_bs,

                             Error **err)

{

    int ret = 0;

    BlockDriverState *bs;

    BdrvNextIterator *it = NULL;

    QEMUSnapshotInfo sn1, *snapshot = &sn1;



    while (ret == 0 && (it = bdrv_next(it, &bs))) {

        AioContext *ctx = bdrv_get_aio_context(bs);



        aio_context_acquire(ctx);

        if (bdrv_can_snapshot(bs) &&

                bdrv_snapshot_find(bs, snapshot, name) >= 0) {

            ret = bdrv_snapshot_delete_by_id_or_name(bs, name, err);

        }

        aio_context_release(ctx);

    }



    *first_bad_bs = bs;

    return ret;

}
