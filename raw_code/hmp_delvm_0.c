void hmp_delvm(Monitor *mon, const QDict *qdict)

{

    BlockDriverState *bs;

    Error *err;

    const char *name = qdict_get_str(qdict, "name");



    if (!find_vmstate_bs()) {

        monitor_printf(mon, "No block device supports snapshots\n");

        return;

    }



    if (bdrv_all_delete_snapshot(name, &bs, &err) < 0) {

        monitor_printf(mon,

                       "Error while deleting snapshot on device '%s': %s\n",

                       bdrv_get_device_name(bs), error_get_pretty(err));

        error_free(err);

    }

}
