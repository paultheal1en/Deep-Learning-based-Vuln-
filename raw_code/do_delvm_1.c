void do_delvm(Monitor *mon, const QDict *qdict)

{

    DriveInfo *dinfo;

    BlockDriverState *bs, *bs1;

    int ret;

    const char *name = qdict_get_str(qdict, "name");



    bs = get_bs_snapshots();

    if (!bs) {

        monitor_printf(mon, "No block device supports snapshots\n");

        return;

    }



    QTAILQ_FOREACH(dinfo, &drives, next) {

        bs1 = dinfo->bdrv;

        if (bdrv_has_snapshot(bs1)) {

            ret = bdrv_snapshot_delete(bs1, name);

            if (ret < 0) {

                if (ret == -ENOTSUP)

                    monitor_printf(mon,

                                   "Snapshots not supported on device '%s'\n",

                                   bdrv_get_device_name(bs1));

                else

                    monitor_printf(mon, "Error %d while deleting snapshot on "

                                   "'%s'\n", ret, bdrv_get_device_name(bs1));

            }

        }

    }

}
