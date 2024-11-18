void tcp_start_outgoing_migration(MigrationState *s,

                                  const char *host_port,

                                  Error **errp)

{

    Error *err = NULL;

    SocketAddress *saddr = tcp_build_address(host_port, &err);

    if (!err) {

        socket_start_outgoing_migration(s, saddr, &err);

    }

    error_propagate(errp, err);

}
