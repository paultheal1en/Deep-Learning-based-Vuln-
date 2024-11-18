static void migrate_fd_cleanup(MigrationState *s)

{

    int ret = 0;



    if (s->file) {

        DPRINTF("closing file\n");

        ret = qemu_fclose(s->file);

        s->file = NULL;

    }



    assert(s->fd == -1);

    if (ret < 0 && s->state == MIG_STATE_ACTIVE) {

        s->state = MIG_STATE_ERROR;

    }



    if (s->state != MIG_STATE_ACTIVE) {

        qemu_savevm_state_cancel();

    }

}
