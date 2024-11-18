static int net_socket_connect_init(VLANState *vlan,

                                   const char *model,

                                   const char *name,

                                   const char *host_str)

{

    NetSocketState *s;

    int fd, connected, ret, err;

    struct sockaddr_in saddr;



    if (parse_host_port(&saddr, host_str) < 0)

        return -1;



    fd = socket(PF_INET, SOCK_STREAM, 0);

    if (fd < 0) {

        perror("socket");

        return -1;

    }

    socket_set_nonblock(fd);



    connected = 0;

    for(;;) {

        ret = connect(fd, (struct sockaddr *)&saddr, sizeof(saddr));

        if (ret < 0) {

            err = socket_error();

            if (err == EINTR || err == EWOULDBLOCK) {

            } else if (err == EINPROGRESS) {

                break;

#ifdef _WIN32

            } else if (err == WSAEALREADY) {

                break;

#endif

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

    s = net_socket_fd_init(vlan, model, name, fd, connected);

    if (!s)

        return -1;

    snprintf(s->nc.info_str, sizeof(s->nc.info_str),

             "socket: connect to %s:%d",

             inet_ntoa(saddr.sin_addr), ntohs(saddr.sin_port));

    return 0;

}
