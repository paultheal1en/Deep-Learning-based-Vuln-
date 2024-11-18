static int net_socket_listen_init(NetClientState *peer,

                                  const char *model,

                                  const char *name,

                                  const char *host_str)

{

    NetClientState *nc;

    NetSocketState *s;

    SocketAddressLegacy *saddr;

    int ret;

    Error *local_error = NULL;



    saddr = socket_parse(host_str, &local_error);

    if (saddr == NULL) {

        error_report_err(local_error);

        return -1;

    }



    ret = socket_listen(saddr, &local_error);

    if (ret < 0) {

        qapi_free_SocketAddressLegacy(saddr);

        error_report_err(local_error);

        return -1;

    }



    nc = qemu_new_net_client(&net_socket_info, peer, model, name);

    s = DO_UPCAST(NetSocketState, nc, nc);

    s->fd = -1;

    s->listen_fd = ret;

    s->nc.link_down = true;

    net_socket_rs_init(&s->rs, net_socket_rs_finalize);



    qemu_set_fd_handler(s->listen_fd, net_socket_accept, NULL, s);

    qapi_free_SocketAddressLegacy(saddr);

    return 0;

}
