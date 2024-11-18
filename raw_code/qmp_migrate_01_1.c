void qmp_migrate(const char *uri, bool has_blk, bool blk,

                 bool has_inc, bool inc, bool has_detach, bool detach,

                 Error **errp)

{

    Error *local_err = NULL;

    MigrationState *s = migrate_get_current();

    MigrationParams params;

    const char *p;



    params.blk = has_blk && blk;

    params.shared = has_inc && inc;



    if (migration_is_setup_or_active(s->state) ||

        s->state == MIGRATION_STATUS_CANCELLING) {

        error_setg(errp, QERR_MIGRATION_ACTIVE);

        return;

    }

    if (runstate_check(RUN_STATE_INMIGRATE)) {

        error_setg(errp, "Guest is waiting for an incoming migration");

        return;

    }



    if (qemu_savevm_state_blocked(errp)) {

        return;

    }



    if (migration_blockers) {

        *errp = error_copy(migration_blockers->data);

        return;

    }



    s = migrate_init(&params);



    if (strstart(uri, "tcp:", &p)) {

        tcp_start_outgoing_migration(s, p, &local_err);

#ifdef CONFIG_RDMA

    } else if (strstart(uri, "rdma:", &p)) {

        rdma_start_outgoing_migration(s, p, &local_err);

#endif

#if !defined(WIN32)

    } else if (strstart(uri, "exec:", &p)) {

        exec_start_outgoing_migration(s, p, &local_err);

    } else if (strstart(uri, "unix:", &p)) {

        unix_start_outgoing_migration(s, p, &local_err);

    } else if (strstart(uri, "fd:", &p)) {

        fd_start_outgoing_migration(s, p, &local_err);

#endif

    } else {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, "uri",

                   "a valid migration protocol");

        migrate_set_state(&s->state, MIGRATION_STATUS_SETUP,

                          MIGRATION_STATUS_FAILED);

        return;

    }



    if (local_err) {

        migrate_fd_error(s);

        error_propagate(errp, local_err);

        return;

    }

}
