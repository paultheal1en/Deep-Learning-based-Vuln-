void migrate_fd_error(MigrationState *s, const Error *error)

{

    trace_migrate_fd_error(error ? error_get_pretty(error) : "");

    assert(s->to_dst_file == NULL);

    migrate_set_state(&s->state, MIGRATION_STATUS_SETUP,

                      MIGRATION_STATUS_FAILED);

    if (!s->error) {

        s->error = error_copy(error);

    }

    notifier_list_notify(&migration_state_notifiers, s);

}
