QemuOpts *qemu_opts_from_qdict(QemuOptsList *list, const QDict *qdict)

{

    QemuOpts *opts;



    opts = qemu_opts_create(list, qdict_get_try_str(qdict, "id"), 1);

    if (opts == NULL)

        return NULL;



    qdict_iter(qdict, qemu_opts_from_qdict_1, opts);

    return opts;

}
