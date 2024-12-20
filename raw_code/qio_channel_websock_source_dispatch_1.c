qio_channel_websock_source_dispatch(GSource *source,

                                    GSourceFunc callback,

                                    gpointer user_data)

{

    QIOChannelFunc func = (QIOChannelFunc)callback;

    QIOChannelWebsockSource *wsource = (QIOChannelWebsockSource *)source;

    GIOCondition cond = 0;



    if (wsource->wioc->rawinput.offset) {

        cond |= G_IO_IN;

    }

    if (wsource->wioc->rawoutput.offset < QIO_CHANNEL_WEBSOCK_MAX_BUFFER) {

        cond |= G_IO_OUT;

    }



    return (*func)(QIO_CHANNEL(wsource->wioc),

                   (cond & wsource->condition),

                   user_data);

}
