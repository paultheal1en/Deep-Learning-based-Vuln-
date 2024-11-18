static SocketAddress *unix_build_address(const char *path)

{

    SocketAddress *saddr;



    saddr = g_new0(SocketAddress, 1);

    saddr->type = SOCKET_ADDRESS_KIND_UNIX;

    saddr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

    saddr->u.q_unix.data->path = g_strdup(path);



    return saddr;

}
