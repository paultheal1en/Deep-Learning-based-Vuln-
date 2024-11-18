static void migrate_fd_cancel(MigrationState *s)

{

    if (s->state != MIG_STATE_ACTIVE)

        return;



    DPRINTF("cancelling migration\n");



    s->state = MIG_STATE_CANCELLED;

    notifier_list_notify(&migration_state_notifiers, s);



    migrate_fd_cleanup(s);

}
