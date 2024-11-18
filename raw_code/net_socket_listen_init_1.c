static int net_socket_listen_init(NetClientState *peer,

                                  const char *model,

                                  const char *name,

                                  const char *host_str)

{

    NetClientState *nc;

    NetSocketState *s;

    struct sockaddr_in saddr;

    int fd, ret;

    Error *err = NULL;



    if (parse_host_port(&saddr, host_str, &err) < 0) {

        error_report_err(err);

        return -1;

    }



    fd = qemu_socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0) {

        perror("socket");

        return -1;

    }

    qemu_set_nonblock(fd);



    socket_set_fast_reuse(fd);



    ret = bind(fd, (struct sockaddr *)&saddr, sizeof(saddr));

    if (ret < 0) {

        perror("bind");

        closesocket(fd);

        return -1;

    }

    ret = listen(fd, 0);

    if (ret < 0) {

        perror("listen");

        closesocket(fd);

        return -1;

    }



    nc = qemu_new_net_client(&net_socket_info, peer, model, name);

    s = DO_UPCAST(NetSocketState, nc, nc);

    s->fd = -1;

    s->listen_fd = fd;

    s->nc.link_down = true;

    net_socket_rs_init(&s->rs, net_socket_rs_finalize, false);



    qemu_set_fd_handler(s->listen_fd, net_socket_accept, NULL, s);

    return 0;

}
