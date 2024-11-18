static void external_snapshot_prepare(BlkTransactionStates *common,

                                      Error **errp)

{

    BlockDriver *proto_drv;

    BlockDriver *drv;

    int flags, ret;

    Error *local_err = NULL;

    const char *device;

    const char *new_image_file;

    const char *format = "qcow2";

    enum NewImageMode mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    ExternalSnapshotStates *states =

                             DO_UPCAST(ExternalSnapshotStates, common, common);

    TransactionAction *action = common->action;



    /* get parameters */

    g_assert(action->kind == TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_SYNC);



    device = action->blockdev_snapshot_sync->device;

    new_image_file = action->blockdev_snapshot_sync->snapshot_file;

    if (action->blockdev_snapshot_sync->has_format) {

        format = action->blockdev_snapshot_sync->format;

    }

    if (action->blockdev_snapshot_sync->has_mode) {

        mode = action->blockdev_snapshot_sync->mode;

    }



    /* start processing */

    drv = bdrv_find_format(format);

    if (!drv) {

        error_set(errp, QERR_INVALID_BLOCK_FORMAT, format);

        return;

    }



    states->old_bs = bdrv_find(device);

    if (!states->old_bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    if (!bdrv_is_inserted(states->old_bs)) {

        error_set(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        return;

    }



    if (bdrv_in_use(states->old_bs)) {

        error_set(errp, QERR_DEVICE_IN_USE, device);

        return;

    }



    if (!bdrv_is_read_only(states->old_bs)) {

        if (bdrv_flush(states->old_bs)) {

            error_set(errp, QERR_IO_ERROR);

            return;

        }

    }



    flags = states->old_bs->open_flags;



    proto_drv = bdrv_find_protocol(new_image_file);

    if (!proto_drv) {

        error_set(errp, QERR_INVALID_BLOCK_FORMAT, format);

        return;

    }



    /* create new image w/backing file */

    if (mode != NEW_IMAGE_MODE_EXISTING) {

        bdrv_img_create(new_image_file, format,

                        states->old_bs->filename,

                        states->old_bs->drv->format_name,

                        NULL, -1, flags, &local_err, false);

        if (error_is_set(&local_err)) {

            error_propagate(errp, local_err);

            return;

        }

    }



    /* We will manually add the backing_hd field to the bs later */

    states->new_bs = bdrv_new("");

    /* TODO Inherit bs->options or only take explicit options with an

     * extended QMP command? */

    ret = bdrv_open(states->new_bs, new_image_file, NULL,

                    flags | BDRV_O_NO_BACKING, drv);

    if (ret != 0) {

        error_setg_file_open(errp, -ret, new_image_file);

    }

}
