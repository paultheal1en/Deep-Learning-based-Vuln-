void tcp_start_outgoing_migration(MigrationState *s, const char *host_port, Error **errp)

{

    inet_nonblocking_connect(host_port, tcp_wait_for_connect, s, errp);

}
