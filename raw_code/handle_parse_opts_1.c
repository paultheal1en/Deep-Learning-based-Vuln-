static int handle_parse_opts(QemuOpts *opts, FsDriverEntry *fse, Error **errp)
{
    const char *sec_model = qemu_opt_get(opts, "security_model");
    const char *path = qemu_opt_get(opts, "path");
    if (sec_model) {
        error_report("Invalid argument security_model specified with handle fsdriver");
        return -1;
    }
    if (!path) {
        error_report("fsdev: No path specified");
        return -1;
    }
    fse->path = g_strdup(path);
    return 0;
}