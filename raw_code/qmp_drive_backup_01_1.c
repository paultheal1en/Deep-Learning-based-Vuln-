void qmp_drive_backup(const char *device, const char *target,

                      bool has_format, const char *format,

                      enum MirrorSyncMode sync,

                      bool has_mode, enum NewImageMode mode,

                      bool has_speed, int64_t speed,

                      bool has_on_source_error, BlockdevOnError on_source_error,

                      bool has_on_target_error, BlockdevOnError on_target_error,

                      Error **errp)

{

    BlockDriverState *bs;

    BlockDriverState *target_bs;

    BlockDriverState *source = NULL;

    AioContext *aio_context;

    BlockDriver *drv = NULL;

    Error *local_err = NULL;

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



    bs = bdrv_find(device);

    if (!bs) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }



    aio_context = bdrv_get_aio_context(bs);

    aio_context_acquire(aio_context);



    /* Although backup_run has this check too, we need to use bs->drv below, so

     * do an early check redundantly. */

    if (!bdrv_is_inserted(bs)) {

        error_set(errp, QERR_DEVICE_HAS_NO_MEDIUM, device);

        goto out;

    }



    if (!has_format) {

        format = mode == NEW_IMAGE_MODE_EXISTING ? NULL : bs->drv->format_name;

    }

    if (format) {

        drv = bdrv_find_format(format);

        if (!drv) {

            error_set(errp, QERR_INVALID_BLOCK_FORMAT, format);

            goto out;

        }

    }




    if (bdrv_op_is_blocked(bs, BLOCK_OP_TYPE_BACKUP_SOURCE, errp)) {

        goto out;

    }



    flags = bs->open_flags | BDRV_O_RDWR;



    /* See if we have a backing HD we can use to create our new image

     * on top of. */

    if (sync == MIRROR_SYNC_MODE_TOP) {

        source = bs->backing_hd;

        if (!source) {

            sync = MIRROR_SYNC_MODE_FULL;

        }

    }

    if (sync == MIRROR_SYNC_MODE_NONE) {

        source = bs;

    }



    size = bdrv_getlength(bs);

    if (size < 0) {

        error_setg_errno(errp, -size, "bdrv_getlength failed");

        goto out;

    }



    if (mode != NEW_IMAGE_MODE_EXISTING) {

        assert(format && drv);

        if (source) {

            bdrv_img_create(target, format, source->filename,

                            source->drv->format_name, NULL,

                            size, flags, &local_err, false);

        } else {

            bdrv_img_create(target, format, NULL, NULL, NULL,

                            size, flags, &local_err, false);

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        goto out;

    }



    target_bs = NULL;

    ret = bdrv_open(&target_bs, target, NULL, NULL, flags, drv, &local_err);

    if (ret < 0) {

        error_propagate(errp, local_err);

        goto out;

    }



    bdrv_set_aio_context(target_bs, aio_context);



    backup_start(bs, target_bs, speed, sync, on_source_error, on_target_error,

                 block_job_cb, bs, &local_err);

    if (local_err != NULL) {

        bdrv_unref(target_bs);

        error_propagate(errp, local_err);

        goto out;

    }



out:

    aio_context_release(aio_context);

}