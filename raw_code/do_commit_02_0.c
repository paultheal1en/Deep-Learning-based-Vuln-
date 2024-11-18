void do_commit(Monitor *mon, const QDict *qdict)

{

    const char *device = qdict_get_str(qdict, "device");

    BlockDriverState *bs;



    if (!strcmp(device, "all")) {

        bdrv_commit_all();

    } else {

        bs = bdrv_find(device);

        if (!bs) {

            qerror_report(QERR_DEVICE_NOT_FOUND, device);

            return;

        }

        bdrv_commit(bs);

    }

}
