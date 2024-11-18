static void migrate_fd_cancel(MigrationState *s)
{
    int old_state ;
    QEMUFile *f = migrate_get_current()->to_dst_file;
    trace_migrate_fd_cancel();
    if (s->rp_state.from_dst_file) {
        /* shutdown the rp socket, so causing the rp thread to shutdown */
        qemu_file_shutdown(s->rp_state.from_dst_file);
    do {
        old_state = s->state;
        if (!migration_is_setup_or_active(old_state)) {
            break;
        migrate_set_state(&s->state, old_state, MIGRATION_STATUS_CANCELLING);
    } while (s->state != MIGRATION_STATUS_CANCELLING);
    /*
     * If we're unlucky the migration code might be stuck somewhere in a
     * send/write while the network has failed and is waiting to timeout;
     * if we've got shutdown(2) available then we can force it to quit.
     * The outgoing qemu file gets closed in migrate_fd_cleanup that is
     * called in a bh, so there is no race against this cancel.
     */
    if (s->state == MIGRATION_STATUS_CANCELLING && f) {
        qemu_file_shutdown(f);