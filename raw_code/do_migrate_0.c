int do_migrate(Monitor *mon, const QDict *qdict, QObject **ret_data)

{

    MigrationState *s = migrate_get_current();

    const char *p;

    int detach = qdict_get_try_bool(qdict, "detach", 0);

    int blk = qdict_get_try_bool(qdict, "blk", 0);

    int inc = qdict_get_try_bool(qdict, "inc", 0);

    const char *uri = qdict_get_str(qdict, "uri");

    int ret;



    if (s->state == MIG_STATE_ACTIVE) {

        monitor_printf(mon, "migration already in progress\n");

        return -1;

    }



    if (qemu_savevm_state_blocked(mon)) {

        return -1;

    }



    if (migration_blockers) {

        Error *err = migration_blockers->data;

        qerror_report_err(err);

        return -1;

    }



    s = migrate_init(mon, detach, blk, inc);



    if (strstart(uri, "tcp:", &p)) {

        ret = tcp_start_outgoing_migration(s, p);

#if !defined(WIN32)

    } else if (strstart(uri, "exec:", &p)) {

        ret = exec_start_outgoing_migration(s, p);

    } else if (strstart(uri, "unix:", &p)) {

        ret = unix_start_outgoing_migration(s, p);

    } else if (strstart(uri, "fd:", &p)) {

        ret = fd_start_outgoing_migration(s, p);

#endif

    } else {

        monitor_printf(mon, "unknown migration protocol: %s\n", uri);

        ret  = -EINVAL;

    }



    if (ret < 0) {

        monitor_printf(mon, "migration failed: %s\n", strerror(-ret));

        return ret;

    }



    if (detach) {

        s->mon = NULL;

    }



    notifier_list_notify(&migration_state_notifiers, s);

    return 0;

}
