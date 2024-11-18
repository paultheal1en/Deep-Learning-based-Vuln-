static char *SocketAddress_to_str(const char *prefix, SocketAddress *addr,

                                  bool is_listen, bool is_telnet)

{

    switch (addr->type) {

    case SOCKET_ADDRESS_KIND_INET:

        return g_strdup_printf("%s%s:%s:%s%s", prefix,

                               is_telnet ? "telnet" : "tcp",

                               addr->u.inet.data->host,

                               addr->u.inet.data->port,

                               is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_KIND_UNIX:

        return g_strdup_printf("%sunix:%s%s", prefix,

                               addr->u.q_unix.data->path,

                               is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_KIND_FD:

        return g_strdup_printf("%sfd:%s%s", prefix, addr->u.fd.data->str,

                               is_listen ? ",server" : "");

        break;





    default:

        abort();

    }

}