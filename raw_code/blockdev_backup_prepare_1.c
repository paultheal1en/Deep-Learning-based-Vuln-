static void blockdev_backup_prepare(BlkActionState *common, Error **errp)

{

    BlockdevBackupState *state = DO_UPCAST(BlockdevBackupState, common, common);

    BlockdevBackup *backup;

    BlockDriverState *bs, *target;

    Error *local_err = NULL;



    assert(common->action->type == TRANSACTION_ACTION_KIND_BLOCKDEV_BACKUP);

    backup = common->action->u.blockdev_backup.data;



    bs = qmp_get_root_bs(backup->device, errp);

    if (!bs) {

        return;

    }



    target = bdrv_lookup_bs(backup->target, backup->target, errp);

    if (!target) {

        return;

    }



    /* AioContext is released in .clean() */

    state->aio_context = bdrv_get_aio_context(bs);

    if (state->aio_context != bdrv_get_aio_context(target)) {

        state->aio_context = NULL;

        error_setg(errp, "Backup between two IO threads is not implemented");

        return;

    }

    aio_context_acquire(state->aio_context);

    state->bs = bs;

    bdrv_drained_begin(state->bs);



    do_blockdev_backup(backup, common->block_job_txn, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    state->job = state->bs->job;

}
