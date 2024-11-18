static void external_snapshot_prepare(BlkActionState *common,

                                      Error **errp)

{

    int flags = 0, ret;

    QDict *options = NULL;

    Error *local_err = NULL;

    /* Device and node name of the image to generate the snapshot from */

    const char *device;

    const char *node_name;

    /* Reference to the new image (for 'blockdev-snapshot') */

    const char *snapshot_ref;

    /* File name of the new image (for 'blockdev-snapshot-sync') */

    const char *new_image_file;

    ExternalSnapshotState *state =

                             DO_UPCAST(ExternalSnapshotState, common, common);

    TransactionAction *action = common->action;



    /* 'blockdev-snapshot' and 'blockdev-snapshot-sync' have similar

     * purpose but a different set of parameters */

    switch (action->type) {

    case TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT:

        {

            BlockdevSnapshot *s = action->u.blockdev_snapshot;

            device = s->node;

            node_name = s->node;

            new_image_file = NULL;

            snapshot_ref = s->overlay;

        }

        break;

    case TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_SYNC:

        {

            BlockdevSnapshotSync *s = action->u.blockdev_snapshot_sync;

            device = s->has_device ? s->device : NULL;

            node_name = s->has_node_name ? s->node_name : NULL;

            new_image_file = s->snapshot_file;

            snapshot_ref = NULL;

        }

        break;

    default:

        g_assert_not_reached();

    }



    /* start processing */

    if (action_check_completion_mode(common, errp) < 0) {

        return;

    }



    state->old_bs = bdrv_lookup_bs(device, node_name, errp);

    if (!state->old_bs) {

        return;

    }



    /* Acquire AioContext now so any threads operating on old_bs stop */

    state->aio_context = bdrv_get_aio_context(state->old_bs);

    aio_context_acquire(state->aio_context);

    bdrv_drained_begin(state->old_bs);



    if (!bdrv_is_inserted(state->old_bs)) {

        error_setg(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        return;

    }



    if (bdrv_op_is_blocked(state->old_bs,

                           BLOCK_OP_TYPE_EXTERNAL_SNAPSHOT, errp)) {

        return;

    }



    if (!bdrv_is_read_only(state->old_bs)) {

        if (bdrv_flush(state->old_bs)) {

            error_setg(errp, QERR_IO_ERROR);

            return;

        }

    }



    if (!bdrv_is_first_non_filter(state->old_bs)) {

        error_setg(errp, QERR_FEATURE_DISABLED, "snapshot");

        return;

    }



    if (action->type == TRANSACTION_ACTION_KIND_BLOCKDEV_SNAPSHOT_SYNC) {

        BlockdevSnapshotSync *s = action->u.blockdev_snapshot_sync;

        const char *format = s->has_format ? s->format : "qcow2";

        enum NewImageMode mode;

        const char *snapshot_node_name =

            s->has_snapshot_node_name ? s->snapshot_node_name : NULL;



        if (node_name && !snapshot_node_name) {

            error_setg(errp, "New snapshot node name missing");

            return;

        }



        if (snapshot_node_name &&

            bdrv_lookup_bs(snapshot_node_name, snapshot_node_name, NULL)) {

            error_setg(errp, "New snapshot node name already in use");

            return;

        }



        flags = state->old_bs->open_flags;



        /* create new image w/backing file */

        mode = s->has_mode ? s->mode : NEW_IMAGE_MODE_ABSOLUTE_PATHS;

        if (mode != NEW_IMAGE_MODE_EXISTING) {

            int64_t size = bdrv_getlength(state->old_bs);

            if (size < 0) {

                error_setg_errno(errp, -size, "bdrv_getlength failed");

                return;

            }

            bdrv_img_create(new_image_file, format,

                            state->old_bs->filename,

                            state->old_bs->drv->format_name,

                            NULL, size, flags, &local_err, false);

            if (local_err) {

                error_propagate(errp, local_err);

                return;

            }

        }



        options = qdict_new();

        if (s->has_snapshot_node_name) {

            qdict_put(options, "node-name",

                      qstring_from_str(snapshot_node_name));

        }

        qdict_put(options, "driver", qstring_from_str(format));



        flags |= BDRV_O_NO_BACKING;

    }



    assert(state->new_bs == NULL);

    ret = bdrv_open(&state->new_bs, new_image_file, snapshot_ref, options,

                    flags, errp);

    /* We will manually add the backing_hd field to the bs later */

    if (ret != 0) {

        return;

    }



    if (state->new_bs->blk != NULL) {

        error_setg(errp, "The snapshot is already in use by %s",

                   blk_name(state->new_bs->blk));

        return;

    }



    if (bdrv_op_is_blocked(state->new_bs, BLOCK_OP_TYPE_EXTERNAL_SNAPSHOT,

                           errp)) {

        return;

    }



    if (state->new_bs->backing != NULL) {

        error_setg(errp, "The snapshot already has a backing image");

        return;

    }



    if (!state->new_bs->drv->supports_backing) {

        error_setg(errp, "The snapshot does not support backing images");

    }

}
