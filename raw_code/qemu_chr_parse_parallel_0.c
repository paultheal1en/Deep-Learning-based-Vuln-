static void qemu_chr_parse_parallel(QemuOpts *opts, ChardevBackend *backend,

                                    Error **errp)

{

    const char *device = qemu_opt_get(opts, "path");



    if (device == NULL) {

        error_setg(errp, "chardev: parallel: no device path given");

        return;

    }

    backend->parallel = g_new0(ChardevHostdev, 1);

    backend->parallel->device = g_strdup(device);

}
