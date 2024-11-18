int fd_start_outgoing_migration(MigrationState *s, const char *fdname)

{

    s->fd = monitor_get_fd(s->mon, fdname);

    if (s->fd == -1) {

        DPRINTF("fd_migration: invalid file descriptor identifier\n");

        goto err_after_get_fd;

    }



    if (fcntl(s->fd, F_SETFL, O_NONBLOCK) == -1) {

        DPRINTF("Unable to set nonblocking mode on file descriptor\n");

        goto err_after_open;

    }



    s->get_error = fd_errno;

    s->write = fd_write;

    s->close = fd_close;



    migrate_fd_connect(s);

    return 0;



err_after_open:

    close(s->fd);

err_after_get_fd:

    return -1;

}
