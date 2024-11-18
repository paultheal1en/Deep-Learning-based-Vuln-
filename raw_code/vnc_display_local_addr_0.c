char *vnc_display_local_addr(const char *id)

{

    VncDisplay *vs = vnc_display_find(id);

    SocketAddress *addr;

    char *ret;

    Error *err = NULL;



    assert(vs);



    addr = qio_channel_socket_get_local_address(vs->lsock, &err);

    if (!addr) {

        return NULL;

    }



    if (addr->type != SOCKET_ADDRESS_KIND_INET) {

        qapi_free_SocketAddress(addr);

        return NULL;

    }

    ret = g_strdup_printf("%s;%s", addr->u.inet->host, addr->u.inet->port);

    qapi_free_SocketAddress(addr);



    return ret;

}
