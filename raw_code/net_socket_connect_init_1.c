static int net_socket_connect_init(NetClientState *peer,

                                   const char *model,

                                   const char *name,

                                   const char *host_str)

{

    NetSocketState *s;

    int fd, connected, ret;

    struct sockaddr_in saddr;



    if (parse_host_port(&saddr, host_str) < 0)

        return -1;



    fd = qemu_socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0) {

        perror("socket");

        return -1;

    }

    qemu_set_nonblock(fd);



    connected = 0;

    for(;;) {

        ret = connect(fd, (struct sockaddr *)&saddr, sizeof(saddr));

        if (ret < 0) {

            if (errno == EINTR || errno == EWOULDBLOCK) {

                /* continue */

            } else if (errno == EINPROGRESS ||

                       errno == EALREADY ||

                       errno == EINVAL) {

                break;

            } else {

                perror("connect");

                closesocket(fd);

                return -1;

            }

        } else {

            connected = 1;

            break;

        }

    }

    s = net_socket_fd_init(peer, model, name, fd, connected);

    if (!s)

        return -1;

    snprintf(s->nc.info_str, sizeof(s->nc.info_str),

             "socket: connect to %s:%d",

             inet_ntoa(saddr.sin_addr), ntohs(saddr.sin_port));

    return 0;

}
