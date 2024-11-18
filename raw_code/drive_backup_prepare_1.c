static void drive_backup_prepare(BlkActionState *common, Error **errp)

{

    DriveBackupState *state = DO_UPCAST(DriveBackupState, common, common);

    BlockDriverState *bs;

    DriveBackup *backup;

    Error *local_err = NULL;



    assert(common->action->type == TRANSACTION_ACTION_KIND_DRIVE_BACKUP);

    backup = common->action->u.drive_backup.data;



    bs = qmp_get_root_bs(backup->device, errp);

    if (!bs) {

        return;

    }



    /* AioContext is released in .clean() */

    state->aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(state->aio_context);

    bdrv_drained_begin(bs);

    state->bs = bs;



    do_drive_backup(backup, common->block_job_txn, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    state->job = state->bs->job;

}
