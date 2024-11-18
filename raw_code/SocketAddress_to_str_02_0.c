static int SocketAddress_to_str(char *dest, int max_len,

                                const char *prefix, SocketAddress *addr,

                                bool is_listen, bool is_telnet)

{

    switch (addr->type) {

    case SOCKET_ADDRESS_KIND_INET:

        return snprintf(dest, max_len, "%s%s:%s:%s%s", prefix,

                        is_telnet ? "telnet" : "tcp", addr->u.inet->host,

                        addr->u.inet->port, is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_KIND_UNIX:

        return snprintf(dest, max_len, "%sunix:%s%s", prefix,

                        addr->u.q_unix->path, is_listen ? ",server" : "");

        break;

    case SOCKET_ADDRESS_KIND_FD:

        return snprintf(dest, max_len, "%sfd:%s%s", prefix, addr->u.fd->str,

                        is_listen ? ",server" : "");

        break;

    default:

        abort();

    }

}
