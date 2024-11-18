int socket_connect(SocketAddress *addr, Error **errp,

                   NonBlockingConnectHandler *callback, void *opaque)

{

    QemuOpts *opts;

    int fd;



    opts = qemu_opts_create_nofail(&socket_optslist);

    switch (addr->kind) {

    case SOCKET_ADDRESS_KIND_INET:

        inet_addr_to_opts(opts, addr->inet);

        fd = inet_connect_opts(opts, errp, callback, opaque);

        break;



    case SOCKET_ADDRESS_KIND_UNIX:

        qemu_opt_set(opts, "path", addr->q_unix->path);

        fd = unix_connect_opts(opts, errp, callback, opaque);

        break;



    case SOCKET_ADDRESS_KIND_FD:

        fd = monitor_get_fd(cur_mon, addr->fd->str, errp);

        if (callback) {


            callback(fd, opaque);

        }

        break;



    default:

        abort();

    }

    qemu_opts_del(opts);

    return fd;

}