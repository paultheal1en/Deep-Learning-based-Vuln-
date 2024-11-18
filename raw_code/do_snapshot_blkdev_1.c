int do_snapshot_blkdev(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const char *device = qdict_get_str(qdict, "device");

    const char *filename = qdict_get_try_str(qdict, "snapshot_file");

    const char *format = qdict_get_try_str(qdict, "format");

    BlockDriverState *bs;

    BlockDriver *drv, *proto_drv;

    int ret = 0;

    int flags;



    if (!filename) {

        qerror_report(QERR_MISSING_PARAMETER, "snapshot_file");

        ret = -1;

        goto out;

    }



    bs = bdrv_find(device);

    if (!bs) {

        qerror_report(QERR_DEVICE_NOT_FOUND, device);

        ret = -1;

        goto out;

    }



    if (!format) {

        format = "qcow2";

    }



    drv = bdrv_find_format(format);

    if (!drv) {

        qerror_report(QERR_INVALID_BLOCK_FORMAT, format);

        ret = -1;

        goto out;

    }



    proto_drv = bdrv_find_protocol(filename);

    if (!proto_drv) {

        qerror_report(QERR_INVALID_BLOCK_FORMAT, format);

        ret = -1;

        goto out;

    }



    ret = bdrv_img_create(filename, format, bs->filename,

                          bs->drv->format_name, NULL, -1, bs->open_flags);

    if (ret) {

        goto out;

    }



    qemu_aio_flush();

    bdrv_flush(bs);



    flags = bs->open_flags;

    bdrv_close(bs);

    ret = bdrv_open(bs, filename, flags, drv);

    /*

     * If reopening the image file we just created fails, we really

     * are in trouble :(

     */

    if (ret != 0) {

        abort();

    }

out:

    if (ret) {

        ret = -1;

    }



    return ret;

}
