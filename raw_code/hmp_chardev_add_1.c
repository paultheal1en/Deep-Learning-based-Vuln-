void hmp_chardev_add(Monitor *mon, const QDict *qdict)

{

    const char *args = qdict_get_str(qdict, "args");

    Error *err = NULL;

    QemuOpts *opts;



    opts = qemu_opts_parse_noisily(qemu_find_opts("chardev"), args, true);

    if (opts == NULL) {

        error_setg(&err, "Parsing chardev args failed");

    } else {

        qemu_chr_new_from_opts(opts, NULL, &err);


    }

    hmp_handle_error(mon, &err);

}