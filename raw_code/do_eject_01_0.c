int do_eject(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    BlockDriverState *bs;

    int force = qdict_get_int(qdict, "force");

    const char *filename = qdict_get_str(qdict, "device");



    bs = bdrv_find(filename);

    if (!bs) {

        qerror_report(QERR_DEVICE_NOT_FOUND, filename);

        return -1;

    }

    return eject_device(mon, bs, force);

}
