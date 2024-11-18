static CharDriverState *qmp_chardev_open_socket(ChardevSocket *sock,

                                                Error **errp)

{

    SocketAddress *addr = sock->addr;

    bool do_nodelay     = sock->has_nodelay ? sock->nodelay : false;

    bool is_listen      = sock->has_server  ? sock->server  : true;

    bool is_telnet      = sock->has_telnet  ? sock->telnet  : false;

    bool is_waitconnect = sock->has_wait    ? sock->wait    : false;

    int fd;



    if (is_listen) {

        fd = socket_listen(addr, errp);

    } else {

        fd = socket_connect(addr, errp, NULL, NULL);

    }

    if (error_is_set(errp)) {

        return NULL;

    }

    return qemu_chr_open_socket_fd(fd, do_nodelay, is_listen,

                                   is_telnet, is_waitconnect, errp);

}
