static void fd_put_buffer(void *opaque, const uint8_t *buf,

                          int64_t pos, int size)

{

    QEMUFileFD *s = opaque;

    ssize_t len;



    do {

        len = write(s->fd, buf, size);

    } while (len == -1 && errno == EINTR);



    if (len == -1)

        len = -errno;



    /* When the fd becomes writable again, register a callback to do

     * a put notify */

    if (len == -EAGAIN)

        qemu_set_fd_handler2(s->fd, NULL, NULL, fd_put_notify, s);

}
