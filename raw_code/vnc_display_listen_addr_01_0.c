static int vnc_display_listen_addr(VncDisplay *vd,

                                   SocketAddress *addr,

                                   const char *name,

                                   QIOChannelSocket ***lsock,

                                   guint **lsock_tag,

                                   size_t *nlsock,

                                   Error **errp)

{

    *nlsock = 1;

    *lsock = g_new0(QIOChannelSocket *, 1);

    *lsock_tag = g_new0(guint, 1);



    (*lsock)[0] = qio_channel_socket_new();

    qio_channel_set_name(QIO_CHANNEL((*lsock)[0]), name);

    if (qio_channel_socket_listen_sync((*lsock)[0], addr, errp) < 0) {

        return -1;

    }



    (*lsock_tag)[0] = qio_channel_add_watch(

        QIO_CHANNEL((*lsock)[0]),

        G_IO_IN, vnc_listen_io, vd, NULL);



    return 0;

}
