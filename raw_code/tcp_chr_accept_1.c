static void tcp_chr_accept(void *opaque)

{

    CharDriverState *chr = opaque;

    TCPCharDriver *s = chr->opaque;

    struct sockaddr_in saddr;

#ifndef _WIN32

    struct sockaddr_un uaddr;

#endif

    struct sockaddr *addr;

    socklen_t len;

    int fd;



    for(;;) {

#ifndef _WIN32

	if (s->is_unix) {

	    len = sizeof(uaddr);

	    addr = (struct sockaddr *)&uaddr;

	} else

#endif

	{

	    len = sizeof(saddr);

	    addr = (struct sockaddr *)&saddr;

	}

        fd = accept(s->listen_fd, addr, &len);

        if (fd < 0 && errno != EINTR) {

            return;

        } else if (fd >= 0) {

            if (s->do_telnetopt)

                tcp_chr_telnet_init(fd);

            break;

        }

    }

    socket_set_nonblock(fd);

    if (s->do_nodelay)

        socket_set_nodelay(fd);

    s->fd = fd;

    qemu_set_fd_handler(s->listen_fd, NULL, NULL, NULL);

    tcp_chr_connect(chr);

}
