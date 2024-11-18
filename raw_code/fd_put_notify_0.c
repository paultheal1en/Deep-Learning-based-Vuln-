static void fd_put_notify(void *opaque)

{

    QEMUFileFD *s = opaque;



    /* Remove writable callback and do a put notify */

    qemu_set_fd_handler2(s->fd, NULL, NULL, NULL, NULL);

    qemu_file_put_notify(s->file);

}
