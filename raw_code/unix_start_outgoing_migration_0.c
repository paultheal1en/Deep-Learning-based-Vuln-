void unix_start_outgoing_migration(MigrationState *s,

                                   const char *path,

                                   Error **errp)

{

    SocketAddressLegacy *saddr = unix_build_address(path);

    socket_start_outgoing_migration(s, saddr, errp);

}
