void unix_start_outgoing_migration(MigrationState *s, const char *path, Error **errp)

{

    unix_nonblocking_connect(path, unix_wait_for_connect, s, errp);

}
