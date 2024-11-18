static int qemu_chr_open_win_file_out(QemuOpts *opts, CharDriverState **_chr)

{

    const char *file_out = qemu_opt_get(opts, "path");

    HANDLE fd_out;



    fd_out = CreateFile(file_out, GENERIC_WRITE, FILE_SHARE_READ, NULL,

                        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (fd_out == INVALID_HANDLE_VALUE) {

        return -EIO;

    }



    return qemu_chr_open_win_file(fd_out, _chr);

}
