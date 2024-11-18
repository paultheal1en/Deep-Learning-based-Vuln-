static void external_snapshot_prepare(BlkTransactionState *common,

                                      Error **errp)

{

    BlockDriver *drv;

    int flags, ret;

    QDict *options = NULL;

    Error *local_err = NULL;

    bool has_device = false;

    const char *device;

    bool has_node_name = false;

    const char *node_name;

    bool has_snapshot_node_name = false;

    const char *snapshot_node_name;

    const char *new_image_file;

    const char *format = "qcow2";

    enum NewImageMode mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    ExternalSnapshotState *state =

                             DO_UPCAST(ExternalSnapshotState, common, common);

    TransactionAction *action = common->action;



    /* get parameters */

    g_assert(action->kind == TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_SYNC);



    has_device = action->blockdev_snapshot_sync->has_device;

    device = action->blockdev_snapshot_sync->device;

    has_node_name = action->blockdev_snapshot_sync->has_node_name;

    node_name = action->blockdev_snapshot_sync->node_name;

    has_snapshot_node_name =

        action->blockdev_snapshot_sync->has_snapshot_node_name;

    snapshot_node_name = action->blockdev_snapshot_sync->snapshot_node_name;



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



    state->old_bs = bdrv_lookup_bs(has_device ? device : NULL,

                                   has_node_name ? node_name : NULL,

                                   &local_err);

    if (error_is_set(&local_err)) {

        error_propagate(errp, local_err);

        return;

    }



    if (has_node_name && !has_snapshot_node_name) {

        error_setg(errp, "New snapshot node name missing");

        return;

    }



    if (has_snapshot_node_name && bdrv_find_node(snapshot_node_name)) {

        error_setg(errp, "New snapshot node name already existing");

        return;

    }



    if (!bdrv_is_inserted(state->old_bs)) {

        error_set(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        return;

    }



    if (bdrv_in_use(state->old_bs)) {

        error_set(errp, QERR_DEVICE_IN_USE, device);

        return;

    }



    if (!bdrv_is_read_only(state->old_bs)) {

        if (bdrv_flush(state->old_bs)) {

            error_set(errp, QERR_IO_ERROR);

            return;

        }

    }



    if (!bdrv_is_first_non_filter(state->old_bs)) {

        error_set(errp, QERR_FEATURE_DISABLED, "snapshot");

        return;

    }



    flags = state->old_bs->open_flags;



    /* create new image w/backing file */

    if (mode != NEW_IMAGE_MODE_EXISTING) {

        bdrv_img_create(new_image_file, format,

                        state->old_bs->filename,

                        state->old_bs->drv->format_name,

                        NULL, -1, flags, &local_err, false);

        if (error_is_set(&local_err)) {

            error_propagate(errp, local_err);

            return;

        }

    }



    if (has_snapshot_node_name) {

        options = qdict_new();

        qdict_put(options, "node-name",

                  qstring_from_str(snapshot_node_name));

    }



    /* We will manually add the backing_hd field to the bs later */

    state->new_bs = bdrv_new("");

    /* TODO Inherit bs->options or only take explicit options with an

     * extended QMP command? */

    ret = bdrv_open(state->new_bs, new_image_file, options,

                    flags | BDRV_O_NO_BACKING, drv, &local_err);

    if (ret != 0) {

        error_propagate(errp, local_err);

    }



    QDECREF(options);

}
