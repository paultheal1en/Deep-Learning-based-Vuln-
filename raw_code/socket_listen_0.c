int socket_listen(SocketAddressLegacy *addr, Error **errp)

{

    int fd;



    switch (addr->type) {

    case SOCKET_ADDRESS_LEGACY_KIND_INET:

        fd = inet_listen_saddr(addr->u.inet.data, 0, false, errp);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_UNIX:

        fd = unix_listen_saddr(addr->u.q_unix.data, false, errp);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_FD:

        fd = monitor_get_fd(cur_mon, addr->u.fd.data->str, errp);

        break;



    case SOCKET_ADDRESS_LEGACY_KIND_VSOCK:

        fd = vsock_listen_saddr(addr->u.vsock.data, errp);

        break;



    default:

        abort();

    }

    return fd;

}
