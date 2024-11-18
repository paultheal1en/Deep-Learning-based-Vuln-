void backup_start(BlockDriverState *bs, BlockDriverState *target,
                  int64_t speed, MirrorSyncMode sync_mode,
                  BlockdevOnError on_source_error,
                  BlockdevOnError on_target_error,
                  BlockCompletionFunc *cb, void *opaque,
                  Error **errp)
{
    int64_t len;
    assert(bs);
    assert(target);
    assert(cb);
    if (bs == target) {
        error_setg(errp, "Source and target cannot be the same");
    if ((on_source_error == BLOCKDEV_ON_ERROR_STOP ||
         on_source_error == BLOCKDEV_ON_ERROR_ENOSPC) &&
        !bdrv_iostatus_is_enabled(bs)) {
        error_set(errp, QERR_INVALID_PARAMETER, "on-source-error");
    len = bdrv_getlength(bs);
    if (len < 0) {
        error_setg_errno(errp, -len, "unable to get length for '%s'",
    BackupBlockJob *job = block_job_create(&backup_job_driver, bs, speed,
                                           cb, opaque, errp);
    if (!job) {
    bdrv_op_block_all(target, job->common.blocker);
    job->on_source_error = on_source_error;
    job->on_target_error = on_target_error;
    job->target = target;
    job->sync_mode = sync_mode;
    job->common.len = len;
    job->common.co = qemu_coroutine_create(backup_run);
    qemu_coroutine_enter(job->common.co, job);