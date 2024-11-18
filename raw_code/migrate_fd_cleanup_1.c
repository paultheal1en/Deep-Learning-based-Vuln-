static void migrate_fd_cleanup(void *opaque)

{

    MigrationState *s = opaque;



    qemu_bh_delete(s->cleanup_bh);

    s->cleanup_bh = NULL;



    if (s->file) {

        trace_migrate_fd_cleanup();

        qemu_mutex_unlock_iothread();

        qemu_thread_join(&s->thread);

        qemu_mutex_lock_iothread();



        qemu_fclose(s->file);

        s->file = NULL;

    }



    assert(s->state != MIG_STATE_ACTIVE);



    if (s->state != MIG_STATE_COMPLETED) {

        qemu_savevm_state_cancel();

        if (s->state == MIG_STATE_CANCELLING) {

            migrate_set_state(s, MIG_STATE_CANCELLING, MIG_STATE_CANCELLED);

        }

    }



    notifier_list_notify(&migration_state_notifiers, s);

}
