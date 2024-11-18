void do_blockdev_backup(const char *job_id, const char *device,

                        const char *target, enum MirrorSyncMode sync,

                         bool has_speed, int64_t speed,

                         bool has_on_source_error,

                         BlockdevOnError on_source_error,

                         bool has_on_target_error,

                         BlockdevOnError on_target_error,

                         BlockJobTxn *txn, Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;

    BlockDriverState *target_bs;

    Error *local_err = NULL;

    AioContext *aio_context;



    if (!has_speed) {

        speed = 0;

    }

    if (!has_on_source_error) {

        on_source_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_on_target_error) {

        on_target_error = BLOCKDEV_ON_ERROR_REPORT;

    }



    blk = blk_by_name(device);

    if (!blk) {

        error_setg(errp, "Device '%s' not found", device);

        return;

    }



    aio_context = blk_get_aio_context(blk);

    aio_context_acquire(aio_context);



    if (!blk_is_available(blk)) {

        error_setg(errp, "Device '%s' has no medium", device);

        goto out;

    }

    bs = blk_bs(blk);



    target_bs = bdrv_lookup_bs(target, target, errp);

    if (!target_bs) {

        goto out;

    }



    if (bdrv_get_aio_context(target_bs) != aio_context) {

        if (!bdrv_has_blk(target_bs)) {

            /* The target BDS is not attached, we can safely move it to another

             * AioContext. */

            bdrv_set_aio_context(target_bs, aio_context);

        } else {

            error_setg(errp, "Target is attached to a different thread from "

                             "source.");

            goto out;

        }

    }

    backup_start(job_id, bs, target_bs, speed, sync, NULL, on_source_error,

                 on_target_error, block_job_cb, bs, txn, &local_err);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

    }

out:

    aio_context_release(aio_context);

}
