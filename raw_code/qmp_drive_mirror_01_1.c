void qmp_drive_mirror(const char *device, const char *target,

                      bool has_format, const char *format,

                      enum MirrorSyncMode sync,

                      bool has_mode, enum NewImageMode mode,

                      bool has_speed, int64_t speed, Error **errp)

{

    BlockDriverInfo bdi;

    BlockDriverState *bs;

    BlockDriverState *source, *target_bs;

    BlockDriver *proto_drv;

    BlockDriver *drv = NULL;

    Error *local_err = NULL;

    int flags;

    uint64_t size;

    int ret;



    if (!has_speed) {

        speed = 0;

    }

    if (!has_mode) {

        mode = NEW_IMAGE_MODE_ABSOLUTE_PATHS;

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



    if (bdrv_in_use(bs)) {

        error_set(errp, QERR_DEVICE_IN_USE, device);

        return;

    }



    flags = bs->open_flags | BDRV_O_RDWR;

    source = bs->backing_hd;

    if (!source && sync == MIRROR_SYNC_MODE_TOP) {

        sync = MIRROR_SYNC_MODE_FULL;

    }



    proto_drv = bdrv_find_protocol(target);

    if (!proto_drv) {

        error_set(errp, QERR_INVALID_BLOCK_FORMAT, format);

        return;

    }



    if (sync == MIRROR_SYNC_MODE_FULL && mode != NEW_IMAGE_MODE_EXISTING) {

        /* create new image w/o backing file */

        assert(format && drv);

        bdrv_get_geometry(bs, &size);

        size *= 512;

        ret = bdrv_img_create(target, format,

                              NULL, NULL, NULL, size, flags);

    } else {

        switch (mode) {

        case NEW_IMAGE_MODE_EXISTING:

            ret = 0;

            break;

        case NEW_IMAGE_MODE_ABSOLUTE_PATHS:

            /* create new image with backing file */

            ret = bdrv_img_create(target, format,

                                  source->filename,

                                  source->drv->format_name,

                                  NULL, -1, flags);

            break;

        default:

            abort();

        }

    }



    if (ret) {

        error_set(errp, QERR_OPEN_FILE_FAILED, target);

        return;

    }



    target_bs = bdrv_new("");

    ret = bdrv_open(target_bs, target, flags | BDRV_O_NO_BACKING, drv);



    if (ret < 0) {

        bdrv_delete(target_bs);

        error_set(errp, QERR_OPEN_FILE_FAILED, target);

        return;

    }



    /* We need a backing file if we will copy parts of a cluster.  */

    if (bdrv_get_info(target_bs, &bdi) >= 0 && bdi.cluster_size != 0 &&

        bdi.cluster_size >= BDRV_SECTORS_PER_DIRTY_CHUNK * 512) {

        ret = bdrv_open_backing_file(target_bs);

        if (ret < 0) {

            bdrv_delete(target_bs);

            error_set(errp, QERR_OPEN_FILE_FAILED, target);

            return;

        }

    }



    mirror_start(bs, target_bs, speed, sync, block_job_cb, bs, &local_err);

    if (local_err != NULL) {

        bdrv_delete(target_bs);

        error_propagate(errp, local_err);

        return;

    }



    /* Grab a reference so hotplug does not delete the BlockDriverState from

     * underneath us.

     */

    drive_get_ref(drive_get_by_blockdev(bs));

}
