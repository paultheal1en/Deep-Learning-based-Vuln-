static int local_parse_opts(QemuOpts *opts, struct FsDriverEntry *fse)

{

    const char *sec_model = qemu_opt_get(opts, "security_model");

    const char *path = qemu_opt_get(opts, "path");



    if (!sec_model) {

        fprintf(stderr, "security model not specified, "

                "local fs needs security model\nvalid options are:"

                "\tsecurity_model=[passthrough|mapped|none]\n");

        return -1;

    }



    if (!strcmp(sec_model, "passthrough")) {

        fse->export_flags |= V9FS_SM_PASSTHROUGH;

    } else if (!strcmp(sec_model, "mapped")) {

        fse->export_flags |= V9FS_SM_MAPPED;

    } else if (!strcmp(sec_model, "none")) {

        fse->export_flags |= V9FS_SM_NONE;

    } else {

        fprintf(stderr, "Invalid security model %s specified, valid options are"

                "\n\t [passthrough|mapped|none]\n", sec_model);

        return -1;

    }



    if (!path) {

        fprintf(stderr, "fsdev: No path specified.\n");

        return -1;

    }

    fse->path = g_strdup(path);



    return 0;

}
