static CharDriverState *qemu_chr_open_tcp(const char *host_str,

                                          int is_telnet,

					  int is_unix)

{

    CharDriverState *chr = NULL;

    TCPCharDriver *s = NULL;

    int fd = -1, ret, err, val;

    int is_listen = 0;

    int is_waitconnect = 1;

    int do_nodelay = 0;

    const char *ptr;

    struct sockaddr_in saddr;

#ifndef _WIN32

    struct sockaddr_un uaddr;

#endif

    struct sockaddr *addr;

    socklen_t addrlen;



#ifndef _WIN32

    if (is_unix) {

	addr = (struct sockaddr *)&uaddr;

	addrlen = sizeof(uaddr);

	if (parse_unix_path(&uaddr, host_str) < 0)

	    goto fail;

    } else

#endif

    {

	addr = (struct sockaddr *)&saddr;

	addrlen = sizeof(saddr);

	if (parse_host_port(&saddr, host_str) < 0)

	    goto fail;

    }



    ptr = host_str;

    while((ptr = strchr(ptr,','))) {

        ptr++;

        if (!strncmp(ptr,"server",6)) {

            is_listen = 1;

        } else if (!strncmp(ptr,"nowait",6)) {

            is_waitconnect = 0;

        } else if (!strncmp(ptr,"nodelay",6)) {

            do_nodelay = 1;

        } else {

            printf("Unknown option: %s\n", ptr);

            goto fail;

        }

    }

    if (!is_listen)

        is_waitconnect = 0;



    chr = qemu_mallocz(sizeof(CharDriverState));

    if (!chr)

        goto fail;

    s = qemu_mallocz(sizeof(TCPCharDriver));

    if (!s)

        goto fail;



#ifndef _WIN32

    if (is_unix)

	fd = socket(PF_UNIX, SOCK_STREAM, 0);

    else

#endif

	fd = socket(PF_INET, SOCK_STREAM, 0);



    if (fd < 0)

        goto fail;



    if (!is_waitconnect)

        socket_set_nonblock(fd);



    s->connected = 0;

    s->fd = -1;

    s->listen_fd = -1;

    s->is_unix = is_unix;

    s->do_nodelay = do_nodelay && !is_unix;



    chr->opaque = s;

    chr->chr_write = tcp_chr_write;

    chr->chr_close = tcp_chr_close;



    if (is_listen) {

        /* allow fast reuse */

#ifndef _WIN32

	if (is_unix) {

	    char path[109];

	    pstrcpy(path, sizeof(path), uaddr.sun_path);

	    unlink(path);

	} else

#endif

	{

	    val = 1;

	    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&val, sizeof(val));

	}



        ret = bind(fd, addr, addrlen);

        if (ret < 0)

            goto fail;



        ret = listen(fd, 0);

        if (ret < 0)

            goto fail;



        s->listen_fd = fd;

        qemu_set_fd_handler(s->listen_fd, tcp_chr_accept, NULL, chr);

        if (is_telnet)

            s->do_telnetopt = 1;

    } else {

        for(;;) {

            ret = connect(fd, addr, addrlen);

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

                    goto fail;

                }

            } else {

                s->connected = 1;

                break;

            }

        }

        s->fd = fd;

        socket_set_nodelay(fd);

        if (s->connected)

            tcp_chr_connect(chr);

        else

            qemu_set_fd_handler(s->fd, NULL, tcp_chr_connect, chr);

    }



    if (is_listen && is_waitconnect) {

        printf("QEMU waiting for connection on: %s\n", host_str);

        tcp_chr_accept(chr);

        socket_set_nonblock(s->listen_fd);

    }



    return chr;

 fail:

    if (fd >= 0)

        closesocket(fd);

    qemu_free(s);

    qemu_free(chr);

    return NULL;

}
