int do_drive_del(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    const char *id = qdict_get_str(qdict, "id");

    BlockDriverState *bs;



    bs = bdrv_find(id);

    if (!bs) {

        qerror_report(QERR_DEVICE_NOT_FOUND, id);

        return -1;

    }

    if (bdrv_in_use(bs)) {

        qerror_report(QERR_DEVICE_IN_USE, id);

        return -1;

    }



    /* quiesce block driver; prevent further io */

    qemu_aio_flush();

    bdrv_flush(bs);

    bdrv_close(bs);



    /* if we have a device attached to this BlockDriverState

     * then we need to make the drive anonymous until the device

     * can be removed.  If this is a drive with no device backing

     * then we can just get rid of the block driver state right here.

     */

    if (bdrv_get_attached_dev(bs)) {

        bdrv_make_anon(bs);

    } else {

        drive_uninit(drive_get_by_blockdev(bs));

    }



    return 0;

}
