static int migrate_fd_cleanup(MigrationState *s)

{

    int ret = 0;



    qemu_set_fd_handler2(s->fd, NULL, NULL, NULL, NULL);



    if (s->file) {

        DPRINTF("closing file\n");

        if (qemu_fclose(s->file) != 0) {

            ret = -1;

        }

        s->file = NULL;

    } else {

        if (s->mon) {

            monitor_resume(s->mon);

        }

    }



    if (s->fd != -1) {

        close(s->fd);

        s->fd = -1;

    }



    return ret;

}
