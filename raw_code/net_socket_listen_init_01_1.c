static int net_socket_listen_init(VLANState *vlan,

                                  const char *model,

                                  const char *name,

                                  const char *host_str)

{

    NetSocketListenState *s;

    int fd, val, ret;

    struct sockaddr_in saddr;



    if (parse_host_port(&saddr, host_str) < 0)

        return -1;



    s = g_malloc0(sizeof(NetSocketListenState));



    fd = qemu_socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0) {

        perror("socket");

        g_free(s);

        return -1;

    }

    socket_set_nonblock(fd);



    /* allow fast reuse */

    val = 1;

    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&val, sizeof(val));



    ret = bind(fd, (struct sockaddr *)&saddr, sizeof(saddr));

    if (ret < 0) {

        perror("bind");

        g_free(s);


        return -1;

    }

    ret = listen(fd, 0);

    if (ret < 0) {

        perror("listen");

        g_free(s);


        return -1;

    }

    s->vlan = vlan;

    s->model = g_strdup(model);

    s->name = name ? g_strdup(name) : NULL;

    s->fd = fd;

    qemu_set_fd_handler(fd, net_socket_accept, NULL, s);

    return 0;

}