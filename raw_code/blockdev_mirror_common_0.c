static void blockdev_mirror_common(BlockDriverState *bs,

                                   BlockDriverState *target,

                                   bool has_replaces, const char *replaces,

                                   enum MirrorSyncMode sync,

                                   bool has_speed, int64_t speed,

                                   bool has_granularity, uint32_t granularity,

                                   bool has_buf_size, int64_t buf_size,

                                   bool has_on_source_error,

                                   BlockdevOnError on_source_error,

                                   bool has_on_target_error,

                                   BlockdevOnError on_target_error,

                                   bool has_unmap, bool unmap,

                                   Error **errp)

{



    if (!has_speed) {

        speed = 0;

    }

    if (!has_on_source_error) {

        on_source_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_on_target_error) {

        on_target_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_granularity) {

        granularity = 0;

    }

    if (!has_buf_size) {

        buf_size = 0;

    }

    if (!has_unmap) {

        unmap = true;

    }



    if (granularity != 0 && (granularity < 512 || granularity > 1048576 * 64)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "granularity",

                   "a value in range [512B, 64MB]");

        return;

    }

    if (granularity & (granularity - 1)) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "granularity",

                   "power of 2");

        return;

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_MIRROR_SOURCE, errp)) {

        return;

    }

    if (bdrv_op_is_blocked(target, BLOCK_OP_TYPE_MIRROR_TARGET, errp)) {

        return;

    }

    if (target->blk) {

        error_setg(errp, "Cannot mirror to an attached block device");

        return;

    }



    if (!bs->backing && sync == MIRROR_SYNC_MODE_TOP) {

        sync = MIRROR_SYNC_MODE_FULL;

    }



    /* pass the node name to replace to mirror start since it's loose coupling

     * and will allow to check whether the node still exist at mirror completion

     */

    mirror_start(bs, target,

                 has_replaces ? replaces : NULL,

                 speed, granularity, buf_size, sync,

                 on_source_error, on_target_error, unmap,

                 block_job_cb, bs, errp);

}
