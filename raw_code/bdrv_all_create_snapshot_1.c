int bdrv_all_create_snapshot(QEMUSnapshotInfo *sn,

                             BlockDriverState *vm_state_bs,

                             uint64_t vm_state_size,

                             BlockDriverState **first_bad_bs)

{

    int err = 0;

    BlockDriverState *bs;

    BdrvNextIterator it;



    for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

        AioContext *ctx = bdrv_get_aio_context(bs);



        aio_context_acquire(ctx);

        if (bs == vm_state_bs) {

            sn->vm_state_size = vm_state_size;

            err = bdrv_snapshot_create(bs, sn);

        } else if (bdrv_can_snapshot(bs)) {

            sn->vm_state_size = 0;

            err = bdrv_snapshot_create(bs, sn);

        }

        aio_context_release(ctx);

        if (err < 0) {


            goto fail;

        }

    }



fail:

    *first_bad_bs = bs;

    return err;

}