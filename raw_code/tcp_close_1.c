static int tcp_close(MigrationState *s)

{

    DPRINTF("tcp_close\n");

    if (s->fd != -1) {

        close(s->fd);

        s->fd = -1;

    }

    return 0;

}
