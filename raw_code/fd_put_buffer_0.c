static ssize_t fd_put_buffer(void *opaque, const void *data, size_t size)

{

    FdMigrationState *s = opaque;

    ssize_t ret;



    do {

        ret = write(s->fd, data, size);

    } while (ret == -1 && errno == EINTR);



    if (ret == -1)

        ret = -errno;



    if (ret == -EAGAIN)

        qemu_set_fd_handler2(s->fd, NULL, NULL, fd_put_notify, s);



    return ret;

}
