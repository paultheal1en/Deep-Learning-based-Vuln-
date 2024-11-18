static int net_socket_udp_init(NetClientState *peer,

                                 const char *model,

                                 const char *name,

                                 const char *rhost,

                                 const char *lhost)

{

    NetSocketState *s;

    int fd, ret;

    struct sockaddr_in laddr, raddr;



    if (parse_host_port(&laddr, lhost) < 0) {

        return -1;

    }



    if (parse_host_port(&raddr, rhost) < 0) {

        return -1;

    }



    fd = qemu_socket(PF_INET, SOCK_DGRAM, 0);

    if (fd < 0) {

        perror("socket(PF_INET, SOCK_DGRAM)");

        return -1;

    }



    ret = socket_set_fast_reuse(fd);

    if (ret < 0) {

        closesocket(fd);

        return -1;

    }

    ret = bind(fd, (struct sockaddr *)&laddr, sizeof(laddr));

    if (ret < 0) {

        perror("bind");

        closesocket(fd);

        return -1;

    }

    qemu_set_nonblock(fd);



    s = net_socket_fd_init(peer, model, name, fd, 0);

    if (!s) {

        return -1;

    }



    s->dgram_dst = raddr;



    snprintf(s->nc.info_str, sizeof(s->nc.info_str),

             "socket: udp=%s:%d",

             inet_ntoa(raddr.sin_addr), ntohs(raddr.sin_port));

    return 0;

}
