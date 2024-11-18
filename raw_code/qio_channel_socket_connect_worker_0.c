static void qio_channel_socket_connect_worker(QIOTask *task,

                                              gpointer opaque)

{

    QIOChannelSocket *ioc = QIO_CHANNEL_SOCKET(qio_task_get_source(task));

    SocketAddress *addr = opaque;

    Error *err = NULL;



    qio_channel_socket_connect_sync(ioc, addr, &err);



    qio_task_set_error(task, err);

}
