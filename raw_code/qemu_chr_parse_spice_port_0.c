static void qemu_chr_parse_spice_port(QemuOpts *opts, ChardevBackend *backend,

                                      Error **errp)

{

    const char *name = qemu_opt_get(opts, "name");

    ChardevSpicePort *spiceport;



    if (name == NULL) {

        error_setg(errp, "chardev: spice port: no name given");

        return;

    }

    spiceport = backend->u.spiceport = g_new0(ChardevSpicePort, 1);

    qemu_chr_parse_common(opts, qapi_ChardevSpicePort_base(spiceport));

    spiceport->fqdn = g_strdup(name);

}
