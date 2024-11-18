static int migrate_fd_close(void *opaque)

{

    MigrationState *s = opaque;



    if (s->mon) {

        monitor_resume(s->mon);

    }

    qemu_set_fd_handler2(s->fd, NULL, NULL, NULL, NULL);

    return s->close(s);

}
