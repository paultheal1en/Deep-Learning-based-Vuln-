static gboolean qio_channel_tls_handshake_io(QIOChannel *ioc,

                                             GIOCondition condition,

                                             gpointer user_data)

{

    QIOTask *task = user_data;

    QIOChannelTLS *tioc = QIO_CHANNEL_TLS(

        qio_task_get_source(task));



    qio_channel_tls_handshake_task(

       tioc, task);



    object_unref(OBJECT(tioc));



    return FALSE;

}
