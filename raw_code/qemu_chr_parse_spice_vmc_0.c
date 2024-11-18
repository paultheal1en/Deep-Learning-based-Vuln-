static void qemu_chr_parse_spice_vmc(QemuOpts *opts, ChardevBackend *backend,

                                     Error **errp)

{

    const char *name = qemu_opt_get(opts, "name");



    if (name == NULL) {

        error_setg(errp, "chardev: spice channel: no name given");

        return;

    }

    backend->u.spicevmc = g_new0(ChardevSpiceChannel, 1);

    backend->u.spicevmc->type = g_strdup(name);

}
