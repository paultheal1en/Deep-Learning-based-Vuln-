static CharDriverState *qmp_chardev_open_serial(ChardevHostdev *serial,

                                                Error **errp)

{

#ifdef HAVE_CHARDEV_TTY

    int fd;



    fd = qmp_chardev_open_file_source(serial->device, O_RDWR, errp);

    if (error_is_set(errp)) {

        return NULL;

    }

    qemu_set_nonblock(fd);

    return qemu_chr_open_tty_fd(fd);

#else

    error_setg(errp, "character device backend type 'serial' not supported");

    return NULL;

#endif

}
