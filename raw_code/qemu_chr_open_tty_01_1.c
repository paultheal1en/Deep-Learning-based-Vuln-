static int qemu_chr_open_tty(QemuOpts *opts, CharDriverState **_chr)

{

    const char *filename = qemu_opt_get(opts, "path");

    CharDriverState *chr;

    int fd;



    TFR(fd = qemu_open(filename, O_RDWR | O_NONBLOCK));

    if (fd < 0) {

        return -errno;

    }

    tty_serial_init(fd, 115200, 'N', 8, 1);

    chr = qemu_chr_open_fd(fd, fd);

    chr->chr_ioctl = tty_serial_ioctl;

    chr->chr_close = qemu_chr_close_tty;



    *_chr = chr;

    return 0;

}
