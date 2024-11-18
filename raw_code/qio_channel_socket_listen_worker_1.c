static int qio_channel_socket_listen_worker(QIOTask *task,

                                            Error **errp,

                                            gpointer opaque)

{

    QIOChannelSocket *ioc = QIO_CHANNEL_SOCKET(qio_task_get_source(task));

    SocketAddress *addr = opaque;

    int ret;



    ret = qio_channel_socket_listen_sync(ioc,

                                         addr,

                                         errp);



    object_unref(OBJECT(ioc));

    return ret;

}
