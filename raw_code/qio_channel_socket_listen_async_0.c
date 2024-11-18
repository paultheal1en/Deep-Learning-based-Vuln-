void qio_channel_socket_listen_async(QIOChannelSocket *ioc,

                                     SocketAddress *addr,

                                     QIOTaskFunc callback,

                                     gpointer opaque,

                                     GDestroyNotify destroy)

{

    QIOTask *task = qio_task_new(

        OBJECT(ioc), callback, opaque, destroy);

    SocketAddress *addrCopy;



    addrCopy = QAPI_CLONE(SocketAddress, addr);



    /* socket_listen() blocks in DNS lookups, so we must use a thread */

    trace_qio_channel_socket_listen_async(ioc, addr);

    qio_task_run_in_thread(task,

                           qio_channel_socket_listen_worker,

                           addrCopy,

                           (GDestroyNotify)qapi_free_SocketAddress);

}
