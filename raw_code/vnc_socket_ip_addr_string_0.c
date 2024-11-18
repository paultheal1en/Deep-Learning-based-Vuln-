vnc_socket_ip_addr_string(QIOChannelSocket *ioc,

                          bool local,

                          Error **errp)

{

    SocketAddress *addr;

    char *ret;



    if (local) {

        addr = qio_channel_socket_get_local_address(ioc, errp);

    } else {

        addr = qio_channel_socket_get_remote_address(ioc, errp);

    }

    if (!addr) {

        return NULL;

    }



    if (addr->type != SOCKET_ADDRESS_KIND_INET) {

        error_setg(errp, "Not an inet socket type");

        return NULL;

    }

    ret = g_strdup_printf("%s;%s", addr->u.inet->host, addr->u.inet->port);

    qapi_free_SocketAddress(addr);

    return ret;

}
