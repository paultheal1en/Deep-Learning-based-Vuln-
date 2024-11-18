static int connect_to_sdog(const char *addr, const char *port)

{

    char hbuf[NI_MAXHOST], sbuf[NI_MAXSERV];

    int fd, ret;

    struct addrinfo hints, *res, *res0;



    if (!addr) {

        addr = SD_DEFAULT_ADDR;

        port = SD_DEFAULT_PORT;

    }



    memset(&hints, 0, sizeof(hints));

    hints.ai_socktype = SOCK_STREAM;



    ret = getaddrinfo(addr, port, &hints, &res0);

    if (ret) {

        error_report("unable to get address info %s, %s",

                     addr, strerror(errno));

        return -errno;

    }



    for (res = res0; res; res = res->ai_next) {

        ret = getnameinfo(res->ai_addr, res->ai_addrlen, hbuf, sizeof(hbuf),

                          sbuf, sizeof(sbuf), NI_NUMERICHOST | NI_NUMERICSERV);

        if (ret) {

            continue;

        }



        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);

        if (fd < 0) {

            continue;

        }



    reconnect:

        ret = connect(fd, res->ai_addr, res->ai_addrlen);

        if (ret < 0) {

            if (errno == EINTR) {

                goto reconnect;

            }


            break;

        }



        dprintf("connected to %s:%s\n", addr, port);

        goto success;

    }

    fd = -errno;

    error_report("failed connect to %s:%s", addr, port);

success:

    freeaddrinfo(res0);

    return fd;

}