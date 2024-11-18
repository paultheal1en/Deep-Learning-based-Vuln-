static CharDriverState *qemu_chr_open_file_out(QemuOpts *opts)

{

    int fd_out;



    TFR(fd_out = open(qemu_opt_get(opts, "path"),

                      O_WRONLY | O_TRUNC | O_CREAT | O_BINARY, 0666));

    if (fd_out < 0)

        return NULL;

    return qemu_chr_open_fd(-1, fd_out);

}
