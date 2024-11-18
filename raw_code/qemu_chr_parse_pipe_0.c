static void qemu_chr_parse_pipe(QemuOpts *opts, ChardevBackend *backend,

                                Error **errp)

{

    const char *device = qemu_opt_get(opts, "path");

    ChardevHostdev *dev;



    if (device == NULL) {

        error_setg(errp, "chardev: pipe: no device path given");

        return;

    }

    dev = backend->u.pipe = g_new0(ChardevHostdev, 1);

    qemu_chr_parse_common(opts, qapi_ChardevHostdev_base(dev));

    dev->device = g_strdup(device);

}
