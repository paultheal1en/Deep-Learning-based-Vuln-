static gboolean nbd_accept(QIOChannel *ioc, GIOCondition cond, gpointer opaque)

{

    QIOChannelSocket *cioc;



    cioc = qio_channel_socket_accept(QIO_CHANNEL_SOCKET(ioc),

                                     NULL);

    if (!cioc) {

        return TRUE;

    }



    if (state >= TERMINATE) {

        object_unref(OBJECT(cioc));

        return TRUE;

    }



    nb_fds++;

    nbd_update_server_watch();

    nbd_client_new(newproto ? NULL : exp, cioc,

                   NULL, NULL, nbd_client_closed);

    object_unref(OBJECT(cioc));



    return TRUE;

}
