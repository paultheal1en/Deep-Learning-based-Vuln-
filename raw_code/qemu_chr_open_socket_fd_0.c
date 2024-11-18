static CharDriverState *qemu_chr_open_socket_fd(int fd, bool do_nodelay,

                                                bool is_listen, bool is_telnet,

                                                bool is_waitconnect,

                                                Error **errp)

{

    CharDriverState *chr = NULL;

    TCPCharDriver *s = NULL;

    char host[NI_MAXHOST], serv[NI_MAXSERV];

    const char *left = "", *right = "";

    struct sockaddr_storage ss;

    socklen_t ss_len = sizeof(ss);



    memset(&ss, 0, ss_len);

    if (getsockname(fd, (struct sockaddr *) &ss, &ss_len) != 0) {

        error_setg_errno(errp, errno, "getsockname");

        return NULL;

    }



    chr = g_malloc0(sizeof(CharDriverState));

    s = g_malloc0(sizeof(TCPCharDriver));



    s->connected = 0;

    s->fd = -1;

    s->listen_fd = -1;

    s->read_msgfds = 0;

    s->read_msgfds_num = 0;

    s->write_msgfds = 0;

    s->write_msgfds_num = 0;



    chr->filename = g_malloc(256);

    switch (ss.ss_family) {

#ifndef _WIN32

    case AF_UNIX:

        s->is_unix = 1;

        snprintf(chr->filename, 256, "unix:%s%s",

                 ((struct sockaddr_un *)(&ss))->sun_path,

                 is_listen ? ",server" : "");

        break;

#endif

    case AF_INET6:

        left  = "[";

        right = "]";

        /* fall through */

    case AF_INET:

        s->do_nodelay = do_nodelay;

        getnameinfo((struct sockaddr *) &ss, ss_len, host, sizeof(host),

                    serv, sizeof(serv), NI_NUMERICHOST | NI_NUMERICSERV);

        snprintf(chr->filename, 256, "%s:%s%s%s:%s%s",

                 is_telnet ? "telnet" : "tcp",

                 left, host, right, serv,

                 is_listen ? ",server" : "");

        break;

    }



    chr->opaque = s;

    chr->chr_write = tcp_chr_write;

    chr->chr_sync_read = tcp_chr_sync_read;

    chr->chr_close = tcp_chr_close;

    chr->get_msgfds = tcp_get_msgfds;

    chr->set_msgfds = tcp_set_msgfds;

    chr->chr_add_client = tcp_chr_add_client;

    chr->chr_add_watch = tcp_chr_add_watch;

    chr->chr_update_read_handler = tcp_chr_update_read_handler;

    /* be isn't opened until we get a connection */

    chr->explicit_be_open = true;



    if (is_listen) {

        s->listen_fd = fd;

        s->listen_chan = io_channel_from_socket(s->listen_fd);

        s->listen_tag = g_io_add_watch(s->listen_chan, G_IO_IN, tcp_chr_accept, chr);

        if (is_telnet) {

            s->do_telnetopt = 1;

        }

    } else {

        s->connected = 1;

        s->fd = fd;

        socket_set_nodelay(fd);

        s->chan = io_channel_from_socket(s->fd);

        tcp_chr_connect(chr);

    }



    if (is_listen && is_waitconnect) {

        fprintf(stderr, "QEMU waiting for connection on: %s\n",

                chr->filename);

        tcp_chr_accept(s->listen_chan, G_IO_IN, chr);

        qemu_set_nonblock(s->listen_fd);

    }

    return chr;

}
