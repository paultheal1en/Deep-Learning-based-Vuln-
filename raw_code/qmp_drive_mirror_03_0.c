void qmp_drive_mirror(const char *device, const char *target,

                      bool has_format, const char *format,

                      bool has_node_name, const char *node_name,

                      bool has_replaces, const char *replaces,

                      enum MirrorSyncMode sync,

                      bool has_mode, enum NewImageMode mode,

                      bool has_speed, int64_t speed,

                      bool has_granularity, uint32_t granularity,

                      bool has_buf_size, int64_t buf_size,

                      bool has_on_source_error, BlockdevOnError on_source_error,

                      bool has_on_target_error, BlockdevOnError on_target_error,

                      Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *source, *target_bs;

    BlockDriver *drv = NULL;

    Error *local_err = NULL;

    QDict *options = NULL;

    int flags;

    int64_t size;

    int ret;



    if (!has_speed) {

        speed = 0;

    }

    if (!has_on_source_error) {

        on_source_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_on_target_error) {

        on_target_error = BLOCKDEV_ON_ERROR_REPORT;

    }

    if (!has_mode) {

        mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

    }

    if (!has_granularity) {

        granularity = 0;

    }

    if (!has_buf_size) {

        buf_size = DEFAULT_MIRROR_BUF_SIZE;

    }



    if (granularity != 0 && (granularity < 512 || granularity > 1048576 * 64)) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "granularity",

                  "a value in range [512B, 64MB]");

        return;

    }

    if (granularity & (granularity - 1)) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "granularity", "power of 2");

        return;

    }



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    if (!bdrv_is_inserted(bs)) {

        error_set(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        return;

    }



    if (!has_format) {

        format = mode == NEW_IMAGE_MODE_EXISTING ? NULL : bs->drv->format_name;

    }

    if (format) {

        drv = bdrv_find_format(format);

        if (!drv) {

            error_set(errp, QERR_INVALID_BLOCK_FORMAT, format);

            return;

        }

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_MIRROR, errp)) {

        return;

    }



    flags = bs->open_flags | BDRV_O_RDWR;

    source = bs->backing_hd;

    if (!source && sync == MIRROR_SYNC_MODE_TOP) {

        sync = MIRROR_SYNC_MODE_FULL;

    }

    if (sync == MIRROR_SYNC_MODE_NONE) {

        source = bs;

    }



    size = bdrv_getlength(bs);

    if (size < 0) {

        error_setg_errno(errp, -size, "bdrv_getlength failed");

        return;

    }



    if (has_replaces) {

        BlockDriverState *to_replace_bs;



        if (!has_node_name) {

            error_setg(errp, "a node-name must be provided when replacing a"

                             " named node of the graph");

            return;

        }



        to_replace_bs = check_to_replace_node(replaces, &local_err);



        if (!to_replace_bs) {

            error_propagate(errp, local_err);

            return;

        }



        if (size != bdrv_getlength(to_replace_bs)) {

            error_setg(errp, "cannot replace image with a mirror image of "

                             "different size");

            return;

        }

    }



    if ((sync == MIRROR_SYNC_MODE_FULL || !source)

        && mode != NEW_IMAGE_MODE_EXISTING)

    {

        /* create new image w/o backing file */

        assert(format && drv);

        bdrv_img_create(target, format,

                        NULL, NULL, NULL, size, flags, &local_err, false);

    } else {

        switch (mode) {

        case NEW_IMAGE_MODE_EXISTING:

            break;

        case NEW_IMAGE_MODE_ABSOLUTE_PATHS:

            /* create new image with backing file */

            bdrv_img_create(target, format,

                            source->filename,

                            source->drv->format_name,

                            NULL, size, flags, &local_err, false);

            break;

        default:

            abort();

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (has_node_name) {

        options = qdict_new();

        qdict_put(options, "node-name", qstring_from_str(node_name));

    }



    /* Mirroring takes care of copy-on-write using the source's backing

     * file.

     */

    target_bs = NULL;

    ret = bdrv_open(&target_bs, target, NULL, options,

                    flags | BDRV_O_NO_BACKING, drv, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        return;

    }



    /* pass the node name to replace to mirror start since it's loose coupling

     * and will allow to check whether the node still exist at mirror completion

     */

    mirror_start(bs, target_bs,

                 has_replaces ? replaces : NULL,

                 speed, granularity, buf_size, sync,

                 on_source_error, on_target_error,

                 block_job_cb, bs, &local_err);

    if (local_err != NULL) {

        bdrv_unref(target_bs);

        error_propagate(errp, local_err);

        return;

    }

}
