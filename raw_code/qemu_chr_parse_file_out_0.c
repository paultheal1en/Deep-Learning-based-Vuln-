static void qemu_chr_parse_file_out(QemuOpts *opts, ChardevBackend *backend,

                                    Error **errp)

{

    const char *path = qemu_opt_get(opts, "path");

    ChardevFile *file;



    if (path == NULL) {

        error_setg(errp, "chardev: file: no filename given");

        return;

    }

    file = backend->u.file = g_new0(ChardevFile, 1);

    qemu_chr_parse_common(opts, qapi_ChardevFile_base(file));

    file->out = g_strdup(path);



    file->has_append = true;

    file->append = qemu_opt_get_bool(opts, "append", false);

}
