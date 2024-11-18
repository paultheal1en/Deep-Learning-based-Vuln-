int socket_listen(SocketAddress *addr, Error **errp)

{

    QemuOpts *opts;

    int fd;



    opts = qemu_opts_create(&socket_optslist, NULL, 0, &error_abort);

    switch (addr->kind) {

    case SOCKET_ADDRESS_KIND_INET:

        inet_addr_to_opts(opts, addr->inet);

        fd = inet_listen_opts(opts, 0, errp);

        break;



    case SOCKET_ADDRESS_KIND_UNIX:

        qemu_opt_set(opts, "path", addr->q_unix->path, &error_abort);

        fd = unix_listen_opts(opts, errp);

        break;



    case SOCKET_ADDRESS_KIND_FD:

        fd = monitor_get_fd(cur_mon, addr->fd->str, errp);

        break;



    default:

        abort();

    }

    qemu_opts_del(opts);

    return fd;

}
