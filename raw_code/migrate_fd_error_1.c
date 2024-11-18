void migrate_fd_error(MigrationState *s)

{

    trace_migrate_fd_error();

    assert(s->to_dst_file == NULL);

    migrate_set_state(&s->state, MIGRATION_STATUS_SETUP,

                      MIGRATION_STATUS_FAILED);

    notifier_list_notify(&migration_state_notifiers, s);

}
