void do_delvm(Monitor *mon, const QDict *qdict)

{

    BlockDriverState *bs, *bs1;

    Error *err = NULL;

    const char *name = qdict_get_str(qdict, "name");



    bs = find_vmstate_bs();

    if (!bs) {

        monitor_printf(mon, "No block device supports snapshots\n");

        return;

    }



    bs1 = NULL;

    while ((bs1 = bdrv_next(bs1))) {

        if (bdrv_can_snapshot(bs1)) {

            bdrv_snapshot_delete_by_id_or_name(bs, name, &err);

            if (err) {

                monitor_printf(mon,

                               "Error while deleting snapshot on device '%s':"

                               " %s\n",

                               bdrv_get_device_name(bs),

                               error_get_pretty(err));

                error_free(err);

            }

        }

    }

}
