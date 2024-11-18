void fd_start_incoming_migration(const char *infd, Error **errp)

{

    int fd;

    QEMUFile *f;



    DPRINTF("Attempting to start an incoming migration via fd\n");



    fd = strtol(infd, NULL, 0);

    f = qemu_fdopen(fd, "rb");

    if(f == NULL) {

        error_setg_errno(errp, errno, "failed to open the source descriptor");

        return;

    }



    qemu_set_fd_handler2(fd, NULL, fd_accept_incoming_migration, NULL, f);

}
