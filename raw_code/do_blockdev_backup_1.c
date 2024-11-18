void do_blockdev_backup(BlockdevBackup *backup, BlockJobTxn *txn, Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *target_bs;

    Error *local_err = NULL;

    AioContext *aio_context;



    if (!backup->has_speed) {

        backup->speed = 0;

    }

    if (!backup->has_on_source_error) {

        backup->on_source_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!backup->has_on_target_error) {

        backup->on_target_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!backup->has_job_id) {

        backup->job_id = NULL;

    }

    if (!backup->has_compress) {

        backup->compress = false;

    }



    bs = qmp_get_root_bs(backup->device, errp);

    if (!bs) {

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    target_bs = bdrv_lookup_bs(backup->target, backup->target, errp);

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

    backup_start(backup->job_id, bs, target_bs, backup->speed, backup->sync,

                 NULL, backup->compress, backup->on_source_error,

                 backup->on_target_error, BLOCK_JOB_DEFAULT,

                 NULL, NULL, txn, &local_err);

    if (local_err != NULL) {

        error_propagate(errp, local_err);

    }

out:

    aio_context_release(aio_context);

}
