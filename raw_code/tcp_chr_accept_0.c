static gboolean tcp_chr_accept(QIOChannel *channel,

                               GIOCondition cond,

                               void *opaque)

{

    CharDriverState *chr = opaque;

    TCPCharDriver *s = chr->opaque;

    QIOChannelSocket *sioc;



    sioc = qio_channel_socket_accept(QIO_CHANNEL_SOCKET(channel),

                                     NULL);

    if (!sioc) {

        return TRUE;

    }



    if (s->do_telnetopt) {

        tcp_chr_telnet_init(QIO_CHANNEL(sioc));

    }



    tcp_chr_new_client(chr, sioc);



    object_unref(OBJECT(sioc));



    return TRUE;

}
