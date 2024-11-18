void qmp_change_blockdev(const char *device, const char *filename,

                         const char *format, Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;

    BlockDriver *drv = NULL;

    int bdrv_flags;

    Error *err = NULL;



    blk = blk_by_name(device);

    if (!blk) {

        error_set(errp, QERR_DEVICE_NOT_FOUND, device);

        return;

    }

    bs = blk_bs(blk);



    if (format) {

        drv = bdrv_find_whitelisted_format(format, bs->read_only);

        if (!drv) {

            error_set(errp, QERR_INVALID_BLOCK_FORMAT, format);

            return;

        }

    }



    eject_device(blk, 0, &err);

    if (err) {

        error_propagate(errp, err);

        return;

    }



    bdrv_flags = bdrv_is_read_only(bs) ? 0 : BDRV_O_RDWR;

    bdrv_flags |= bdrv_is_snapshot(bs) ? BDRV_O_SNAPSHOT : 0;



    qmp_bdrv_open_encrypted(bs, filename, bdrv_flags, drv, NULL, errp);

}
