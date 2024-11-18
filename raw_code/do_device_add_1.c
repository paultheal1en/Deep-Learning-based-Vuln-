void do_device_add(Monitor *mon, const QDict *qdict)

{

    QemuOpts *opts;



    opts = qemu_opts_parse(&qemu_device_opts,

                           qdict_get_str(qdict, "config"), "driver");

    if (opts && !qdev_device_help(opts))

        qdev_device_add(opts);

}
