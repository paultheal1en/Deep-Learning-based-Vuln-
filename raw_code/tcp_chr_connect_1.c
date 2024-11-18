static void tcp_chr_connect(void *opaque)

{

    CharDriverState *chr = opaque;

    TCPCharDriver *s = chr->opaque;

    QIOChannelSocket *sioc = QIO_CHANNEL_SOCKET(s->ioc);



    g_free(chr->filename);

    chr->filename = sockaddr_to_str(&sioc->localAddr, sioc->localAddrLen,

                                    &sioc->remoteAddr, sioc->remoteAddrLen,

                                    s->is_listen, s->is_telnet);



    s->connected = 1;

    if (s->ioc) {

        chr->fd_in_tag = io_add_watch_poll(s->ioc,

                                           tcp_chr_read_poll,

                                           tcp_chr_read, chr);

    }

    qemu_chr_be_generic_open(chr);

}
