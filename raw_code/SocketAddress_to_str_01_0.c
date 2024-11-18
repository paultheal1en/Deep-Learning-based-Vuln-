static char *SocketAddress_to_str(const char *prefix, SocketAddressLegacy *addr,

                                  bool is_listen, bool is_telnet)

{

    switch (addr->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        return g_strdup_printf("%s%s:%s:%s%s", prefix,

                               is_telnet ? "telnet" : "tcp",

                               addr->u.inet.data->host,

                               addr->u.inet.data->port,

                               is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_LEGACY_KIND_UNIX:

        return g_strdup_printf("%sunix:%s%s", prefix,

                               addr->u.q_unix.data->path,

                               is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_LEGACY_KIND_FD:

        return g_strdup_printf("%sfd:%s%s", prefix, addr->u.fd.data->str,

                               is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_LEGACY_KIND_VSOCK:

        return g_strdup_printf("%svsock:%s:%s", prefix,

                               addr->u.vsock.data->cid,

                               addr->u.vsock.data->port);

    default:

        abort();

    }

}
