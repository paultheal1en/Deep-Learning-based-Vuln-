int socket_connect(SocketAddress *addr, Error **errp,

                   NonBlockingConnectHandler *callback, void *opaque)

{

    int fd;



    switch (addr->type) {

    case SOCKET_ADDRESS_KIND_INET:

        fd = inet_connect_saddr(addr->u.inet, errp, callback, opaque);

        break;



    case SOCKET_ADDRESS_KIND_UNIX:

        fd = unix_connect_saddr(addr->u.q_unix, errp, callback, opaque);

        break;



    case SOCKET_ADDRESS_KIND_FD:

        fd = monitor_get_fd(cur_mon, addr->u.fd->str, errp);

        if (fd >= 0 && callback) {

            qemu_set_nonblock(fd);

            callback(fd, NULL, opaque);

        }

        break;



    default:

        abort();

    }

    return fd;

}
