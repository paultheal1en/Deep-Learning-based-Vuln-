static void migrate_fd_completed(MigrationState *s)

{

    DPRINTF("setting completed state\n");

    migrate_fd_cleanup(s);

    if (s->state == MIG_STATE_ACTIVE) {

        s->state = MIG_STATE_COMPLETED;

        runstate_set(RUN_STATE_POSTMIGRATE);

    }

    notifier_list_notify(&migration_state_notifiers, s);

}
