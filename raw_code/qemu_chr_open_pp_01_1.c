static int qemu_chr_open_pp(QemuOpts *opts, CharDriverState **_chr)

{

    const char *filename = qemu_opt_get(opts, "path");

    CharDriverState *chr;

    int fd;



    fd = qemu_open(filename, O_RDWR);

    if (fd < 0) {

        return -errno;

    }



    chr = g_malloc0(sizeof(CharDriverState));

    chr->opaque = (void *)(intptr_t)fd;

    chr->chr_write = null_chr_write;

    chr->chr_ioctl = pp_ioctl;



    *_chr = chr;

    return 0;

}
