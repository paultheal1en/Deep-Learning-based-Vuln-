void hmp_drive_add_node(Monitor *mon, const char *optstr)

{

    QemuOpts *opts;

    QDict *qdict;

    Error *local_err = NULL;



    opts = qemu_opts_parse_noisily(&qemu_drive_opts, optstr, false);

    if (!opts) {

        return;

    }



    qdict = qemu_opts_to_qdict(opts, NULL);



    if (!qdict_get_try_str(qdict, "node-name")) {


        error_report("'node-name' needs to be specified");

        goto out;

    }



    BlockDriverState *bs = bds_tree_init(qdict, &local_err);

    if (!bs) {

        error_report_err(local_err);

        goto out;

    }



    QTAILQ_INSERT_TAIL(&monitor_bdrv_states, bs, monitor_list);



out:

    qemu_opts_del(opts);

}