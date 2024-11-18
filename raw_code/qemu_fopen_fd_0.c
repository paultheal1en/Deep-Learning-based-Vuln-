QEMUFile *qemu_fopen_fd(int fd)

{

    QEMUFileFD *s = qemu_mallocz(sizeof(QEMUFileFD));



    if (s == NULL)

        return NULL;



    s->fd = fd;

    s->file = qemu_fopen_ops(s, fd_put_buffer, fd_get_buffer, fd_close, NULL);

    return s->file;

}
