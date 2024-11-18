static void vnc_display_print_local_addr(VncDisplay *vd)

{

    SocketAddressLegacy *addr;

    Error *err = NULL;



    if (!vd->nlsock) {

        return;

    }



    addr = qio_channel_socket_get_local_address(vd->lsock[0], &err);

    if (!addr) {

        return;

    }



    if (addr->type != SOCKET_ADDRESS_LEGACY_KIND_INET) {

        qapi_free_SocketAddressLegacy(addr);

        return;

    }

    error_printf_unless_qmp("VNC server running on %s:%s\n",

                            addr->u.inet.data->host,

                            addr->u.inet.data->port);

    qapi_free_SocketAddressLegacy(addr);

}
