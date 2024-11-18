SocketAddress *socket_address_crumple(SocketAddressFlat *addr_flat)

{

    SocketAddress *addr = g_new(SocketAddress, 1);



    switch (addr_flat->type) {

    case SOCKET_ADDRESS_FLAT_TYPE_INET:

        addr->type = SOCKET_ADDRESS_KIND_INET;

        addr->u.inet.data = QAPI_CLONE(InetSocketAddress,

                                       &addr_flat->u.inet);

        break;

    case SOCKET_ADDRESS_FLAT_TYPE_UNIX:

        addr->type = SOCKET_ADDRESS_KIND_UNIX;

        addr->u.q_unix.data = QAPI_CLONE(UnixSocketAddress,

                                         &addr_flat->u.q_unix);

        break;

    case SOCKET_ADDRESS_FLAT_TYPE_VSOCK:

        addr->type = SOCKET_ADDRESS_KIND_VSOCK;

        addr->u.vsock.data = QAPI_CLONE(VsockSocketAddress,

                                        &addr_flat->u.vsock);

        break;

    case SOCKET_ADDRESS_FLAT_TYPE_FD:

        addr->type = SOCKET_ADDRESS_KIND_FD;

        addr->u.fd.data = QAPI_CLONE(String, &addr_flat->u.fd);

        break;

    default:

        abort();

    }



    return addr;

}
