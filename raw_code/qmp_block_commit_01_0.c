void qmp_block_commit(const char *device,

                      bool has_base, const char *base,

                      bool has_top, const char *top,

                      bool has_backing_file, const char *backing_file,

                      bool has_speed, int64_t speed,

                      Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *base_bs, *top_bs;

    Error *local_err = NULL;

    /* This will be part of the QMP command, if/when the

     * BlockdevOnError change for blkmirror makes it in

     */

    BlockdevOnError on_error = BLOCKDEV_ON_ERROR_REPORT;



    if (!has_speed) {

        speed = 0;

    }



    /* drain all i/o before commits */

    bdrv_drain_all();



    /* Important Note:

     *  libvirt relies on the DeviceNotFound error class in order to probe for

     *  live commit feature versions; for this to work, we must make sure to

     *  perform the device lookup before any generic errors that may occur in a

     *  scenario in which all optional arguments are omitted. */

    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_COMMIT, errp)) {

        return;

    }



    /* default top_bs is the active layer */

    top_bs = bs;



    if (has_top && top) {

        if (strcmp(bs->filename, top) != 0) {

            top_bs = bdrv_find_backing_image(bs, top);

        }

    }



    if (top_bs == NULL) {

        error_setg(errp, "Top image file %s not found", top ? top : "NULL");

        return;

    }



    if (has_base && base) {

        base_bs = bdrv_find_backing_image(top_bs, base);

    } else {

        base_bs = bdrv_find_base(top_bs);

    }



    if (base_bs == NULL) {

        error_set(errp, QERR_BASE_NOT_FOUND, base ? base : "NULL");

        return;

    }



    /* Do not allow attempts to commit an image into itself */

    if (top_bs == base_bs) {

        error_setg(errp, "cannot commit an image into itself");

        return;

    }



    if (top_bs == bs) {

        if (has_backing_file) {

            error_setg(errp, "'backing-file' specified,"

                             " but 'top' is the active layer");

            return;

        }

        commit_active_start(bs, base_bs, speed, on_error, block_job_cb,

                            bs, &local_err);

    } else {

        commit_start(bs, base_bs, top_bs, speed, on_error, block_job_cb, bs,

                     has_backing_file ? backing_file : NULL, &local_err);

    }

    if (local_err != NULL) {

        error_propagate(errp, local_err);

        return;

    }

}
