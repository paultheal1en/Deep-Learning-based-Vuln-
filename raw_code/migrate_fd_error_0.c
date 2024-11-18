void migrate_fd_error(MigrationState *s)

{

    DPRINTF("setting error state\n");

    s->state = MIG_STATE_ERROR;

    notifier_list_notify(&migration_state_notifiers, s);

    migrate_fd_cleanup(s);

}
