void qmp_blockdev_mirror(const char *device, const char *target,

                         bool has_replaces, const char *replaces,

                         MirrorSyncMode sync,

                         bool has_speed, int64_t speed,

                         bool has_granularity, uint32_t granularity,

                         bool has_buf_size, int64_t buf_size,

                         bool has_on_source_error,

                         BlockdevOnError on_source_error,

                         bool has_on_target_error,

                         BlockdevOnError on_target_error,

                         Error **errp)

{

    BlockDriverState *bs;

    BlockBackend *blk;

    BlockDriverState *target_bs;

    AioContext *aio_context;

    BlockMirrorBackingMode backing_mode = MIRROR_LEAVE_BACKING_CHAIN;

    Error *local_err = NULL;



    blk = blk_by_name(device);

    if (!blk) {

        error_setg(errp, "Device '%s' not found", device);

        return;

    }

    bs = blk_bs(blk);



    if (!bs) {

        error_setg(errp, "Device '%s' has no media", device);

        return;

    }



    target_bs = bdrv_lookup_bs(target, target, errp);

    if (!target_bs) {

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    bdrv_set_aio_context(target_bs, aio_context);



    blockdev_mirror_common(bs, target_bs,

                           has_replaces, replaces, sync, backing_mode,

                           has_speed, speed,

                           has_granularity, granularity,

                           has_buf_size, buf_size,

                           has_on_source_error, on_source_error,

                           has_on_target_error, on_target_error,

                           true, true,

                           &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

    }



    aio_context_release(aio_context);

}
