int qemu_opts_set(QemuOptsList *list, const char *id,

                  const char *name, const char *value)

{

    QemuOpts *opts;



    opts = qemu_opts_create(list, id, 1);

    if (opts == NULL) {

        return -1;

    }

    return qemu_opt_set(opts, name, value);

}
