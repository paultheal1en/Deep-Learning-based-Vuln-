static CharDriverState *qmp_chardev_open_file(const char *id,

                                              ChardevBackend *backend,

                                              ChardevReturn *ret,

                                              Error **errp)

{

    ChardevFile *file = backend->u.file;

    ChardevCommon *common = qapi_ChardevFile_base(file);

    HANDLE out;



    if (file->has_in) {

        error_setg(errp, "input file not supported");

        return NULL;

    }



    out = CreateFile(file->out, GENERIC_WRITE, FILE_SHARE_READ, NULL,

                     OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (out == INVALID_HANDLE_VALUE) {

        error_setg(errp, "open %s failed", file->out);

        return NULL;

    }

    return qemu_chr_open_win_file(out, common, errp);

}
