void migrate_fd_connect(MigrationState *s)

{

    s->state = MIG_STATE_SETUP;

    trace_migrate_set_state(MIG_STATE_SETUP);



    /* This is a best 1st approximation. ns to ms */

    s->expected_downtime = max_downtime/1000000;

    s->cleanup_bh = qemu_bh_new(migrate_fd_cleanup, s);



    qemu_file_set_rate_limit(s->file,

                             s->bandwidth_limit / XFER_LIMIT_RATIO);



    qemu_thread_create(&s->thread, migration_thread, s,

                       QEMU_THREAD_JOINABLE);

    notifier_list_notify(&migration_state_notifiers, s);

}
